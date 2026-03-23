"""Training and evaluation for Brahman Steps 1 and 2.

Step 1: S3RNN (σ-only feedback)
Step 2: S3RNNValence (full Hopf decomposition — σ, R, G, B, W)

Trains both models on the same data, evaluates σ separation, tests
autoregressive generation, and compares head-to-head.

Usage:
    python -m brahman.train           # both steps, head-to-head
    python -m brahman.train --step 1  # step 1 only
    python -m brahman.train --step 2  # step 2 only
"""

import argparse
import random
import sys
import time

import torch
import torch.nn.functional as F

from .model import S3RNN, S3RNNValence, S3Transformer
from .data import (
    VOCAB_SIZE, EOS, OPEN, CLOSE,
    make_dataset, pad_batch, corrupt, is_valid, bracket_length,
)


def train_model(
    model,
    name,
    train_data,
    val_data,
    epochs=30,
    batch_size=128,
    lr=1e-3,
    seed=42,
):
    """Train a single model. Returns (model, generation_results)."""
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters())

    print()
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  {name:<58}│")
    print(f"  └─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Parameters:       {n_params:,}")
    print()
    print("  Epoch │ Loss     │ Pred     │ Closure  │ σ mean   │ σ final")
    print("  ──────┼──────────┼──────────┼──────────┼──────────┼──────────")

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_data)

        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_closure = 0.0
        epoch_sigma_mean = 0.0
        epoch_sigma_final = 0.0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch_seqs = train_data[i:i + batch_size]
            padded, total_lengths, b_lengths = pad_batch(batch_seqs)
            tokens = torch.tensor(padded, dtype=torch.long)
            lengths = torch.tensor(b_lengths, dtype=torch.long)

            _, loss, metrics = model(tokens, targets=tokens, lengths=lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pred += metrics["pred"]
            epoch_closure += metrics["closure"]
            epoch_sigma_mean += metrics["sigma_mean"]
            epoch_sigma_final += metrics["sigma_final"]
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches
        epoch_pred /= n_batches
        epoch_closure /= n_batches
        epoch_sigma_mean /= n_batches
        epoch_sigma_final /= n_batches

        print(
            f"  {epoch:>5} │ {epoch_loss:>8.4f} │ {epoch_pred:>8.4f} │ "
            f"{epoch_closure:>8.4f} │ {epoch_sigma_mean:>8.4f} │ "
            f"{epoch_sigma_final:>8.4f}"
        )

    elapsed = time.perf_counter() - t0
    print()
    print(f"  Training complete in {elapsed:.1f}s.")
    print()

    # Evaluate
    sep_result = evaluate(model, val_data, random.Random(seed + 100))
    gen_result = generation_test(model, random.Random(seed + 200))

    return model, sep_result, gen_result


def evaluate(model, val_data, rng):
    """Measure σ separation between valid and corrupted sequences."""
    model.eval()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Evaluation — σ separation                                 │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    valid_sigmas = []
    corrupt_sigmas = []

    with torch.no_grad():
        for seq in val_data:
            bl = bracket_length(seq)

            tokens = torch.tensor([seq], dtype=torch.long)
            _, sigmas = model(tokens)
            valid_sigmas.append(sigmas[0, bl - 1].item())

            c_seq = corrupt(seq, rng=rng)
            c_tokens = torch.tensor([c_seq], dtype=torch.long)
            _, c_sigmas = model(c_tokens)
            corrupt_sigmas.append(c_sigmas[0, bl - 1].item())

    valid_mean = sum(valid_sigmas) / len(valid_sigmas)
    corrupt_mean = sum(corrupt_sigmas) / len(corrupt_sigmas)
    separation = corrupt_mean - valid_mean

    n = len(valid_sigmas)
    valid_var = sum((s - valid_mean) ** 2 for s in valid_sigmas) / (n - 1)
    corrupt_var = sum((s - corrupt_mean) ** 2 for s in corrupt_sigmas) / (n - 1)
    pooled_se = ((valid_var + corrupt_var) / n) ** 0.5
    t_stat = separation / pooled_se if pooled_se > 1e-12 else float("inf")

    correct = sum(1 for v, c in zip(valid_sigmas, corrupt_sigmas) if v < c)
    pair_accuracy = correct / n

    print(f"  σ_final (valid):     {valid_mean:.6f}  (mean over {n} sequences)")
    print(f"  σ_final (corrupted): {corrupt_mean:.6f}")
    print(f"  Separation:          {separation:.6f}")
    print(f"  t-statistic:         {t_stat:.2f}")
    print(f"  Pair accuracy:       {correct}/{n} = {pair_accuracy:.1%}")
    print()

    if t_stat > 2.576:
        print("  Result:  PASS — σ separates valid from corrupted (p < 0.01)")
    elif t_stat > 1.96:
        print("  Result:  MARGINAL — σ separates valid from corrupted (p < 0.05)")
    else:
        print("  Result:  FAIL — no significant σ separation")
    print()

    return {
        "valid_mean": valid_mean,
        "corrupt_mean": corrupt_mean,
        "separation": separation,
        "t_stat": t_stat,
        "pair_accuracy": pair_accuracy,
    }


def generation_test(model, rng, n_sequences=1000, temperature=0.8):
    """Generate sequences autoregressively and count valid ones."""
    model.eval()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Generation test — autoregressive bracket sequences        │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    valid_count = 0
    total_lengths = []
    valid_sigmas = []
    invalid_sigmas = []

    for _ in range(n_sequences):
        tokens, sigmas = model.generate(OPEN, max_length=64, temperature=temperature)
        total_lengths.append(len(tokens))

        bl = bracket_length(tokens)
        final_sigma = sigmas[bl - 1] if 0 < bl <= len(sigmas) else (sigmas[-1] if sigmas else 0.0)

        if is_valid(tokens):
            valid_count += 1
            valid_sigmas.append(final_sigma)
        else:
            invalid_sigmas.append(final_sigma)

    validity_rate = valid_count / n_sequences
    avg_length = sum(total_lengths) / len(total_lengths)

    print(f"  Generated:       {n_sequences} sequences")
    print(f"  Valid:            {valid_count} ({validity_rate:.1%})")
    print(f"  Average length:  {avg_length:.1f} tokens")

    if valid_sigmas:
        print(f"  σ_final (valid):   {sum(valid_sigmas)/len(valid_sigmas):.6f}")
    if invalid_sigmas:
        print(f"  σ_final (invalid): {sum(invalid_sigmas)/len(invalid_sigmas):.6f}")
    print()

    bracket_map = {OPEN: "(", CLOSE: ")", EOS: ""}
    print("  Examples:")
    print()
    for i in range(8):
        tokens, sigmas = model.generate(OPEN, max_length=64, temperature=temperature)
        seq_str = "".join(bracket_map.get(t, "?") for t in tokens)
        valid_tag = "valid" if is_valid(tokens) else "INVALID"
        bl = bracket_length(tokens)
        sigma_val = sigmas[bl - 1] if 0 < bl <= len(sigmas) else 0.0
        print(f"    {i+1}. {seq_str:<44} σ={sigma_val:.4f}  [{valid_tag}]")
    print()

    if validity_rate >= 0.95:
        print("  Result:  PASS — generation validity > 95%")
    elif validity_rate >= 0.80:
        print(f"  Result:  MARGINAL — generation validity {validity_rate:.0%}")
    else:
        print(f"  Result:  FAIL — generation validity {validity_rate:.0%}")
    print()

    return {
        "valid_count": valid_count,
        "total": n_sequences,
        "validity_rate": validity_rate,
        "avg_length": avg_length,
        "valid_sigma": sum(valid_sigmas) / len(valid_sigmas) if valid_sigmas else None,
        "invalid_sigma": sum(invalid_sigmas) / len(invalid_sigmas) if invalid_sigmas else None,
    }


def compare(name_a, result_a, name_b, result_b):
    """Print head-to-head comparison."""
    sep_a, gen_a = result_a
    sep_b, gen_b = result_b

    print()
    print("  ╔═════════════════════════════════════════════════════════════╗")
    print("  ║  Head-to-head comparison                                  ║")
    print("  ╚═════════════════════════════════════════════════════════════╝")
    print()
    print(f"  {'Metric':<30} {'σ-only':<18} {'Valence':<18} {'Winner'}")
    print(f"  {'─' * 30} {'─' * 18} {'─' * 18} {'─' * 10}")

    # σ separation
    t_a, t_b = sep_a["t_stat"], sep_b["t_stat"]
    w = name_b if t_b > t_a else name_a
    print(f"  {'σ separation (t-stat)':<30} {t_a:<18.2f} {t_b:<18.2f} {w}")

    # Pair accuracy
    pa_a, pa_b = sep_a["pair_accuracy"], sep_b["pair_accuracy"]
    w = name_b if pa_b > pa_a else name_a
    print(f"  {'Pair accuracy':<30} {pa_a:<18.1%} {pa_b:<18.1%} {w}")

    # Valid σ (lower is better)
    vs_a = sep_a["valid_mean"]
    vs_b = sep_b["valid_mean"]
    w = name_b if vs_b < vs_a else name_a
    print(f"  {'σ valid (lower = better)':<30} {vs_a:<18.6f} {vs_b:<18.6f} {w}")

    # Generation validity
    gv_a, gv_b = gen_a["validity_rate"], gen_b["validity_rate"]
    w = name_b if gv_b > gv_a else name_a
    print(f"  {'Generation validity':<30} {gv_a:<18.1%} {gv_b:<18.1%} {w}")

    # Average generation length
    gl_a, gl_b = gen_a["avg_length"], gen_b["avg_length"]
    w = name_b if gl_b > gl_a else name_a
    print(f"  {'Avg generation length':<30} {gl_a:<18.1f} {gl_b:<18.1f} {w}")

    # Generation σ valid
    gvs_a = gen_a["valid_sigma"]
    gvs_b = gen_b["valid_sigma"]
    if gvs_a is not None and gvs_b is not None:
        w = name_b if gvs_b < gvs_a else name_a
        print(f"  {'σ generated valid (lower)':<30} {gvs_a:<18.6f} {gvs_b:<18.6f} {w}")

    # Generation σ invalid
    gis_a = gen_a["invalid_sigma"]
    gis_b = gen_b["invalid_sigma"]
    if gis_a is not None and gis_b is not None:
        w = name_b if gis_b > gis_a else name_a
        print(f"  {'σ generated invalid (higher)':<30} {gis_a:<18.6f} {gis_b:<18.6f} {w}")

    print()

    # Verdict
    step2_better = (gv_b >= gv_a) and (t_b >= t_a)
    if step2_better and gv_b > gv_a:
        print("  Verdict:  Valence feedback improves generation.")
    elif gv_b >= gv_a and t_b > t_a:
        print("  Verdict:  Valence feedback improves σ discrimination.")
    elif gv_a == gv_b and abs(t_a - t_b) < 1.0:
        print("  Verdict:  No significant difference between models.")
    else:
        print("  Verdict:  σ-only model performs comparably or better.")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Brahman training")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], default=None,
                        help="Run only step 1, 2, or 3 (default: all)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--closure-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Number of transformer layers (step 3)")
    parser.add_argument("--m-factors", type=int, default=1,
                        help="Number of S³ factors (step 3)")
    args = parser.parse_args(argv)

    n_train = 50_000
    n_val = 2_000

    print()
    print("  ╔═════════════════════════════════════════════════════════════╗")
    print("  ║  Brahman — S³ bracket validation                          ║")
    print("  ╚═════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Vocab:            {VOCAB_SIZE} tokens — ( ) EOS")
    print(f"  Training set:     {n_train:,} sequences (min_pairs=3)")
    print(f"  Validation set:   {n_val:,} sequences")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Hidden dim:       {args.hidden}")
    print(f"  Closure weight:   {args.closure_weight}")

    torch.manual_seed(args.seed)
    train_data = make_dataset(n_train, min_pairs=3, seed=args.seed)
    val_data = make_dataset(n_val, min_pairs=3, seed=args.seed + 1)

    results = {}

    if args.step is None or args.step == 1:
        model1 = S3RNN(VOCAB_SIZE, hidden=args.hidden, closure_weight=args.closure_weight)
        _, sep1, gen1 = train_model(
            model1, "Step 1 — S³ RNN (σ-only)",
            list(train_data), val_data,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
        results["step1"] = (sep1, gen1)

    if args.step is None or args.step == 2:
        model2 = S3RNNValence(VOCAB_SIZE, hidden=args.hidden, closure_weight=args.closure_weight)
        _, sep2, gen2 = train_model(
            model2, "Step 2 — S³ RNN (valence feedback: σ, R, G, B, W)",
            list(train_data), val_data,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
        results["step2"] = (sep2, gen2)

    if args.step is None or args.step == 3:
        model3 = S3Transformer(
            VOCAB_SIZE,
            m_factors=args.m_factors,
            n_layers=args.n_layers,
            hidden=args.hidden,
            closure_weight=args.closure_weight,
        )
        _, sep3, gen3 = train_model(
            model3, f"Step 3 — S³ Transformer ({args.n_layers}L, m={args.m_factors})",
            list(train_data), val_data,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
        results["step3"] = (sep3, gen3)

    # Head-to-head comparisons
    if "step1" in results and "step2" in results:
        compare("σ-only", results["step1"], "Valence", results["step2"])
    if "step1" in results and "step3" in results:
        compare("RNN", results["step1"], "Transformer", results["step3"])
    if "step2" in results and "step3" in results:
        compare("Valence RNN", results["step2"], "Transformer", results["step3"])


if __name__ == "__main__":
    main()
