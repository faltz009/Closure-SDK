"""Training and evaluation for Brahman Visual — grid walk on S³."""

import sys
import os
import time
import math
import random
import argparse

import torch
import torch.nn.functional as F

# Import S3Transformer from the main brahman module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from brahman.model import S3Transformer, qmul

from data import (
    UP, DOWN, LEFT, RIGHT, EOS, VOCAB_SIZE, INVERSES,
    generate_closed_walk, generate_open_walk, generate_dataset,
    walk_to_positions, is_closed, walk_length, pad_batch, walk_to_string,
)


def train_epoch(model, data, optimizer, batch_size, device):
    """One training epoch."""
    model.train()
    random.shuffle(data)

    total_loss = 0.0
    total_pred = 0.0
    total_closure = 0.0
    total_sigma = 0.0
    n_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        padded, lengths = pad_batch(batch)
        tokens = torch.tensor(padded, dtype=torch.long, device=device)
        lens = torch.tensor(lengths, dtype=torch.long, device=device)

        # Lengths for closure: last action position (before EOS)
        action_lens = (lens - 1).clamp(min=1)

        _, loss, metrics = model(tokens, targets=tokens, lengths=action_lens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pred += metrics["pred"]
        total_closure += metrics["closure"]
        total_sigma += metrics["sigma_final"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "pred": total_pred / n_batches,
        "closure": total_closure / n_batches,
        "sigma_final": total_sigma / n_batches,
    }


def evaluate(model, val_data, device, n_samples=2000):
    """σ separation between closed and open walks."""
    model.eval()
    closed_sigmas = []
    open_sigmas = []

    with torch.no_grad():
        # Closed walks (from validation set)
        for seq in val_data[:n_samples]:
            tokens = torch.tensor([seq], dtype=torch.long, device=device)
            _, sigmas = model(tokens)
            wl = walk_length(seq)
            if wl > 0:
                closed_sigmas.append(sigmas[0, wl - 1].item())

        # Open walks (generated fresh)
        for _ in range(n_samples):
            seq = generate_open_walk()
            tokens = torch.tensor([seq], dtype=torch.long, device=device)
            T = tokens.shape[1]
            if T > model.pos_embed.shape[0]:
                continue
            _, sigmas = model(tokens)
            wl = walk_length(seq)
            if wl > 0:
                open_sigmas.append(sigmas[0, wl - 1].item())

    n = min(len(closed_sigmas), len(open_sigmas))
    if n == 0:
        return None

    closed_sigmas = closed_sigmas[:n]
    open_sigmas = open_sigmas[:n]

    closed_mean = sum(closed_sigmas) / n
    open_mean = sum(open_sigmas) / n
    separation = open_mean - closed_mean

    import statistics
    closed_std = statistics.stdev(closed_sigmas) if n > 1 else 1e-12
    open_std = statistics.stdev(open_sigmas) if n > 1 else 1e-12
    pooled_se = math.sqrt(closed_std**2 / n + open_std**2 / n)
    t_stat = separation / pooled_se if pooled_se > 1e-12 else 0.0

    correct = sum(1 for c, o in zip(closed_sigmas, open_sigmas) if c < o)

    return {
        "closed_mean": closed_mean,
        "open_mean": open_mean,
        "separation": separation,
        "t_stat": t_stat,
        "pair_accuracy": correct / n,
        "n": n,
    }


def generation_test(model, device, n_sequences=1000, max_length=32, temperature=0.8):
    """Generate walks and check how many actually close."""
    model.eval()
    valid = 0
    total = 0
    valid_sigmas = []
    invalid_sigmas = []
    examples = []

    with torch.no_grad():
        # Start with a random first action
        for _ in range(n_sequences):
            start_token = random.choice([UP, DOWN, LEFT, RIGHT])
            tokens, sigmas = model.generate(start_token, max_length, temperature, eos_token=EOS)

            # Convert to walk and check closure
            walk = tokens
            closed = is_closed(walk)
            total += 1

            final_sigma = sigmas[-1] if sigmas else 0.0

            if closed:
                valid += 1
                valid_sigmas.append(final_sigma)
            else:
                invalid_sigmas.append(final_sigma)

            if len(examples) < 8:
                examples.append((walk, final_sigma, closed))

    return {
        "valid": valid,
        "total": total,
        "valid_pct": valid / total * 100 if total > 0 else 0,
        "avg_length": sum(walk_length(e[0]) for e in examples) / len(examples) if examples else 0,
        "valid_sigma": sum(valid_sigmas) / len(valid_sigmas) if valid_sigmas else 0,
        "invalid_sigma": sum(invalid_sigmas) / len(invalid_sigmas) if invalid_sigmas else 0,
        "examples": examples,
    }


def check_inverses(model, device):
    """Check if the model discovered that UP/DOWN and LEFT/RIGHT are inverses."""
    model.eval()
    with torch.no_grad():
        # Get embeddings for each action
        tokens = torch.arange(4, dtype=torch.long, device=device).unsqueeze(0)  # [1, 4]
        x = F.one_hot(tokens, VOCAB_SIZE).float()
        x = model.embed(x)
        x = x.view(1, 4, model.m, 4)
        x = F.normalize(x, dim=-1)

        # Extract quaternions: UP=0, DOWN=1, LEFT=2, RIGHT=3
        q_up = x[0, 0]    # [m, 4]
        q_down = x[0, 1]
        q_left = x[0, 2]
        q_right = x[0, 3]

        # Check products: UP·DOWN should ≈ identity, LEFT·RIGHT should ≈ identity
        # Identity on S³ = [1, 0, 0, 0] per factor
        identity = torch.zeros_like(q_up)
        identity[:, 0] = 1.0

        ud = qmul(q_up, q_down)
        ud = F.normalize(ud, dim=-1)
        ud_dist = torch.acos(torch.clamp(ud[:, 0].abs(), max=1 - 1e-7)).mean().item()

        lr = qmul(q_left, q_right)
        lr = F.normalize(lr, dim=-1)
        lr_dist = torch.acos(torch.clamp(lr[:, 0].abs(), max=1 - 1e-7)).mean().item()

        return {
            "up_down_sigma": ud_dist,
            "left_right_sigma": lr_dist,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Brahman Visual — grid walk training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--m-factors", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--closure-weight", type=float, default=0.3)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print()
    print("  ╔═════════════════════════════════════════════════════════════╗")
    print("  ║  Brahman Visual — Grid walk on S³                         ║")
    print("  ╚═════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Vocab:            {VOCAB_SIZE} tokens — ↑ ↓ ← → EOS")
    print(f"  Geometry:         (S³)^{args.m_factors} = {4 * args.m_factors}D")
    print(f"  Layers:           {args.n_layers}")
    print(f"  Training set:     {args.n_train:,} closed walks")
    print(f"  Validation set:   {args.n_val:,} closed walks")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Closure weight:   {args.closure_weight}")
    print()

    # Generate data
    print("  Generating data...", end=" ", flush=True)
    train_data = generate_dataset(args.n_train)
    val_data = generate_dataset(args.n_val)
    avg_len = sum(len(s) for s in train_data) / len(train_data)
    print(f"done. Avg walk length: {avg_len:.1f} tokens")
    print()

    # Show a few examples
    print("  Example walks:")
    for i in range(3):
        s = walk_to_string(train_data[i])
        positions = walk_to_positions(train_data[i])
        print(f"    {s}  →  final={positions[-1]}")
    print()

    # Model
    model = S3Transformer(
        vocab_size=VOCAB_SIZE,
        m_factors=args.m_factors,
        n_layers=args.n_layers,
        hidden=args.hidden,
        closure_weight=args.closure_weight,
        max_seq_len=34,  # max walk = 2*8 + EOS = 17, with padding
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:       {n_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("  Epoch │ Loss     │ Pred     │ Closure  │ σ final  │ Time")
    print("  ──────┼──────────┼──────────┼──────────┼──────────┼──────")

    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        et0 = time.perf_counter()
        metrics = train_epoch(model, train_data, optimizer, args.batch_size, device)
        scheduler.step()
        elapsed = time.perf_counter() - et0

        print(
            f"  {epoch:>5} │ {metrics['loss']:>8.4f} │ {metrics['pred']:>8.4f} │ "
            f"{metrics['closure']:>8.4f} │ {metrics['sigma_final']:>8.4f} │ {elapsed:>5.1f}s"
        )

    total_time = time.perf_counter() - t0
    print()
    print(f"  Training complete in {total_time:.1f}s.")
    print()

    # Inverse check
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Inverse discovery — did the model learn spatial inverses? │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    inv = check_inverses(model, device)
    print(f"  σ(UP · DOWN):     {inv['up_down_sigma']:.6f}  (should be near 0)")
    print(f"  σ(LEFT · RIGHT):  {inv['left_right_sigma']:.6f}  (should be near 0)")
    print()
    if inv['up_down_sigma'] < 0.1 and inv['left_right_sigma'] < 0.1:
        print("  Result:  PASS — model discovered both inverse pairs")
    elif inv['up_down_sigma'] < 0.3 or inv['left_right_sigma'] < 0.3:
        print("  Result:  PARTIAL — model partially discovered inverses")
    else:
        print("  Result:  FAIL — inverses not yet discovered")
    print()

    # σ separation
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Evaluation — σ separation (closed vs open walks)         │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    result = evaluate(model, val_data, device)
    if result:
        print(f"  σ (closed walks):  {result['closed_mean']:.6f}  (mean over {result['n']} walks)")
        print(f"  σ (open walks):    {result['open_mean']:.6f}")
        print(f"  Separation:        {result['separation']:.6f}")
        print(f"  t-statistic:       {result['t_stat']:.2f}")
        print(f"  Pair accuracy:     {result['n'] * result['pair_accuracy']:.0f}/{result['n']} = {result['pair_accuracy']:.1%}")
        print()
        if result['t_stat'] > 2.576:
            print("  Result:  PASS — σ separates closed from open walks (p < 0.01)")
        elif result['t_stat'] > 1.96:
            print("  Result:  MARGINAL — σ separates closed from open walks (p < 0.05)")
        else:
            print("  Result:  FAIL — no significant σ separation")
    print()

    # Generation
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Generation test — autoregressive walk sequences          │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    gen = generation_test(model, device)
    print(f"  Generated:       {gen['total']} walks")
    print(f"  Closed:          {gen['valid']} ({gen['valid_pct']:.1f}%)")
    print(f"  Average length:  {gen['avg_length']:.1f} actions")
    print(f"  σ (closed):      {gen['valid_sigma']:.6f}")
    print(f"  σ (open):        {gen['invalid_sigma']:.6f}")
    print()

    print("  Examples:")
    print()
    for walk, sigma, closed in gen['examples']:
        s = walk_to_string(walk)
        positions = walk_to_positions(walk)
        tag = "closed" if closed else "open"
        print(f"    {s:<40} σ={sigma:.4f}  [{tag}]")
    print()

    if gen['valid_pct'] >= 90:
        print("  Result:  PASS — generation produces closed walks")
    elif gen['valid_pct'] >= 70:
        print("  Result:  MARGINAL — generation partially produces closed walks")
    else:
        print("  Result:  FAIL — generation does not reliably produce closed walks")
    print()


if __name__ == "__main__":
    main()
