"""Drawing Board model — pure geometry, no neural overhead.

The ONLY learned component is the embedding table: V × 4m parameters.
No attention. No prediction head. No cross-entropy.

Forward: sequential Hamilton product composition.
Loss: σ alone (contrastive).
Generation: geodesic nearest neighbor to C⁻¹.

Tests whether stripping neural overhead forces the geometry to
discover compositional inverses — the grid walk experiment showed
the S3Transformer bypasses S³ when attention is available.
"""

import sys
import os
import time
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import qmul
from data import (
    OPEN, CLOSE, EOS, VOCAB_SIZE as BRACKET_VOCAB,
    make_dataset, corrupt, is_valid, bracket_length, pad_batch,
)

# Grid walk tokens
UP, DOWN, LEFT, RIGHT, GRID_EOS = 0, 1, 2, 3, 4
GRID_VOCAB = 5


class S3Pure(nn.Module):
    """Pure geometry model. Embedding table + Hamilton product. Nothing else."""

    def __init__(self, vocab_size, m_factors=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.m = m_factors
        self.dim = 4 * m_factors
        # The ONLY learned parameters
        self.embed = nn.Parameter(torch.randn(vocab_size, m_factors, 4) * 0.3)

    def get_embeddings(self):
        """Unit quaternion embeddings."""
        return F.normalize(self.embed, dim=-1)

    def compose(self, tokens):
        """Sequential composition. Returns (C_final, sigmas_per_step)."""
        B, T = tokens.shape
        device = tokens.device
        e = self.get_embeddings()  # [V, m, 4]

        C = torch.zeros(B, self.m, 4, device=device)
        C[:, :, 0] = 1.0  # identity

        sigmas = []
        for t in range(T):
            g = e[tokens[:, t]]  # [B, m, 4]
            C = F.normalize(qmul(C, g), dim=-1)
            per_factor = torch.acos(torch.clamp(C[:, :, 0].abs(), max=1 - 1e-7))
            sigmas.append(per_factor.mean(dim=-1))  # [B]

        return C, torch.stack(sigmas, dim=1)  # [B, m, 4], [B, T]

    def sigma_at(self, tokens, positions):
        """σ at specific positions per batch element."""
        _, sigmas = self.compose(tokens)
        return sigmas[torch.arange(len(positions)), positions]

    def generate(self, start_token, max_length, eos_token, temperature=0.0):
        """Generate via geodesic nearest neighbor to C⁻¹.

        temperature=0: deterministic (argmin distance).
        temperature>0: sample from softmax(-dist / temperature).
        """
        device = self.embed.device
        e = self.get_embeddings()  # [V, m, 4]

        tokens = [start_token]
        C = torch.zeros(1, self.m, 4, device=device)
        C[:, :, 0] = 1.0

        # Compose start token
        g = e[start_token].unsqueeze(0)
        C = F.normalize(qmul(C, g), dim=-1)

        sigmas = []
        per_factor = torch.acos(torch.clamp(C[:, :, 0].abs(), max=1 - 1e-7))
        sigmas.append(per_factor.mean().item())

        for _ in range(max_length - 1):
            # C⁻¹ = conjugate for unit quaternions
            C_inv = C.clone()
            C_inv[:, :, 1:] = -C_inv[:, :, 1:]

            # Distance from each embedding to C⁻¹
            # dot product per factor, then geodesic distance
            dots = (e.unsqueeze(0) * C_inv).sum(dim=-1)  # [V, m] with broadcast
            dists = torch.acos(torch.clamp(dots.abs(), max=1 - 1e-7))  # [V, m]
            total_dist = dists.sum(dim=-1)  # [V]

            if temperature <= 0:
                next_token = total_dist.argmin().item()
            else:
                logits = -total_dist / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            tokens.append(next_token)

            if next_token == eos_token:
                break

            g = e[next_token].unsqueeze(0)
            C = F.normalize(qmul(C, g), dim=-1)
            per_factor = torch.acos(torch.clamp(C[:, :, 0].abs(), max=1 - 1e-7))
            sigmas.append(per_factor.mean().item())

        return tokens, sigmas


def train_brackets(args):
    """Train pure geometry model on brackets."""
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print()
    print("  ╔═════════════════════════════════════════════════════════════╗")
    print("  ║  Drawing Board — Pure S³ geometry on brackets             ║")
    print("  ╚═════════════════════════════════════════════════════════════╝")
    print()

    model = S3Pure(BRACKET_VOCAB, m_factors=args.m_factors).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Vocab:            {BRACKET_VOCAB} tokens — ( ) EOS")
    print(f"  Geometry:         (S³)^{args.m_factors} = {4 * args.m_factors}D")
    print(f"  Parameters:       {n_params}  (embedding table only)")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Margin:           {args.margin}")
    print()

    # Data
    train_data = make_dataset(args.n_train, seed=args.seed)
    val_data = make_dataset(args.n_val, seed=args.seed + 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("  Epoch │ Loss     │ σ valid  │ σ corrupt │ Time")
    print("  ──────┼──────────┼──────────┼───────────┼──────")

    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        et0 = time.perf_counter()
        model.train()
        random.shuffle(train_data)

        total_loss = 0.0
        total_sv = 0.0
        total_sc = 0.0
        n_batches = 0

        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i:i + args.batch_size]

            # Valid sequences
            padded_v, _, b_lens_v = pad_batch(batch)
            tokens_v = torch.tensor(padded_v, dtype=torch.long, device=device)
            lens_v = torch.tensor(b_lens_v, dtype=torch.long, device=device)

            # Corrupted sequences
            corrupted = [corrupt(s) for s in batch]
            padded_c, _, b_lens_c = pad_batch(corrupted)
            tokens_c = torch.tensor(padded_c, dtype=torch.long, device=device)
            lens_c = torch.tensor(b_lens_c, dtype=torch.long, device=device)

            # Per-step σ: sum across ALL positions (dense signal)
            _, sigmas_v = model.compose(tokens_v)  # [B, T]
            _, sigmas_c = model.compose(tokens_c)

            # Area under σ curve: valid should be smaller than corrupted
            sigma_v = sigmas_v.sum(dim=1).mean()  # total σ area, averaged over batch
            sigma_c = sigmas_c.sum(dim=1).mean()

            # Contrastive loss: push valid area down, corrupted area up
            loss = sigma_v + F.relu(args.margin * sigmas_v.shape[1] - sigma_c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sv += sigma_v.mean().item()
            total_sc += sigma_c.mean().item()
            n_batches += 1

        elapsed = time.perf_counter() - et0
        print(
            f"  {epoch:>5} │ {total_loss / n_batches:>8.4f} │ "
            f"{total_sv / n_batches:>8.4f} │ {total_sc / n_batches:>9.4f} │ {elapsed:>5.1f}s"
        )

    total_time = time.perf_counter() - t0
    print()
    print(f"  Training complete in {total_time:.1f}s.")
    print()

    # ── Inverse check ──
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Inverse discovery — did the model learn ( and ) are       │")
    print("  │  quaternion inverses?                                      │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    model.eval()
    with torch.no_grad():
        e = model.get_embeddings()  # [V, m, 4]
        q_open = e[OPEN]   # [m, 4]
        q_close = e[CLOSE]  # [m, 4]
        q_eos = e[EOS]     # [m, 4]

        # ( · ) should ≈ identity
        product = F.normalize(qmul(q_open, q_close), dim=-1)
        per_factor = torch.acos(torch.clamp(product[:, 0].abs(), max=1 - 1e-7))
        oc_sigma = per_factor.mean().item()

        # EOS should ≈ identity
        eos_sigma = torch.acos(torch.clamp(q_eos[:, 0].abs(), max=1 - 1e-7)).mean().item()

        print(f"  σ( ( · ) ):       {oc_sigma:.6f}  (should be near 0)")
        print(f"  σ(EOS):           {eos_sigma:.6f}  (should be near 0)")
        print()

        if oc_sigma < 0.1:
            print("  Result:  PASS — model discovered ( and ) are inverses")
        elif oc_sigma < 0.3:
            print("  Result:  PARTIAL — model partially discovered inverses")
        else:
            print("  Result:  FAIL — inverses not yet discovered")
        print()

    # ── σ separation ──
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  σ separation (valid vs corrupted brackets)                │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    import statistics
    valid_sigmas = []
    corrupt_sigmas = []

    with torch.no_grad():
        for seq in val_data:
            tokens = torch.tensor([seq], dtype=torch.long, device=device)
            bl = bracket_length(seq)
            if bl > 0:
                _, sigmas = model.compose(tokens)
                valid_sigmas.append(sigmas[0, bl - 1].item())

            c = corrupt(seq)
            tokens_c = torch.tensor([c], dtype=torch.long, device=device)
            bl_c = bracket_length(c)
            if bl_c > 0:
                _, sigmas_c = model.compose(tokens_c)
                corrupt_sigmas.append(sigmas_c[0, bl_c - 1].item())

    n = min(len(valid_sigmas), len(corrupt_sigmas))
    valid_sigmas = valid_sigmas[:n]
    corrupt_sigmas = corrupt_sigmas[:n]

    v_mean = sum(valid_sigmas) / n
    c_mean = sum(corrupt_sigmas) / n
    separation = c_mean - v_mean

    v_std = statistics.stdev(valid_sigmas) if n > 1 else 1e-12
    c_std = statistics.stdev(corrupt_sigmas) if n > 1 else 1e-12
    pooled_se = math.sqrt(v_std**2 / n + c_std**2 / n)
    t_stat = separation / pooled_se if pooled_se > 1e-12 else 0.0

    correct = sum(1 for v, c in zip(valid_sigmas, corrupt_sigmas) if v < c)

    print(f"  σ (valid):         {v_mean:.6f}  (mean over {n})")
    print(f"  σ (corrupted):     {c_mean:.6f}")
    print(f"  Separation:        {separation:.6f}")
    print(f"  t-statistic:       {t_stat:.2f}")
    print(f"  Pair accuracy:     {correct}/{n} = {correct / n:.1%}")
    print()

    if t_stat > 2.576:
        print("  Result:  PASS — σ separates valid from corrupted (p < 0.01)")
    elif t_stat > 1.96:
        print("  Result:  MARGINAL — σ separates (p < 0.05)")
    else:
        print("  Result:  FAIL — no significant σ separation")
    print()

    # ── Generation ──
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Generation — geodesic nearest neighbor to C⁻¹            │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    valid_count = 0
    total_count = 0
    examples = []

    with torch.no_grad():
        for temp in [0.0, 0.3, 0.5]:
            valid_t = 0
            total_t = 0
            examples_t = []
            for _ in range(500):
                tokens, sigmas = model.generate(OPEN, 40, EOS, temperature=temp)
                closed = is_valid(tokens)
                total_t += 1
                if closed:
                    valid_t += 1
                if len(examples_t) < 4:
                    final_s = sigmas[-1] if sigmas else 0.0
                    arrows = {OPEN: "(", CLOSE: ")", EOS: "·"}
                    s = "".join(arrows.get(t, "?") for t in tokens)
                    examples_t.append((s, final_s, closed))

            pct = valid_t / total_t * 100
            label = "deterministic" if temp == 0.0 else f"temp={temp}"
            print(f"  {label:>15}:  {valid_t}/{total_t} = {pct:.1f}% valid")
            for s, sigma, closed in examples_t:
                tag = "valid" if closed else "INVALID"
                print(f"    {s:<40} σ={sigma:.4f}  [{tag}]")
            print()

            if temp == 0.0:
                valid_count = valid_t
                total_count = total_t

    if valid_count / max(total_count, 1) >= 0.9:
        print("  Result:  PASS — generation produces valid brackets")
    elif valid_count / max(total_count, 1) >= 0.7:
        print("  Result:  MARGINAL")
    else:
        print("  Result:  FAIL — generation does not reliably produce valid brackets")
    print()


def train_grid(args):
    """Train pure geometry model on grid walks."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "visual"))
    from data import (
        generate_closed_walk, generate_open_walk, generate_dataset,
        is_closed, walk_length, pad_batch as grid_pad_batch, walk_to_string,
        VOCAB_SIZE as VIS_VOCAB, EOS as VIS_EOS,
    )

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print()
    print("  ╔═════════════════════════════════════════════════════════════╗")
    print("  ║  Drawing Board — Pure S³ geometry on grid walks           ║")
    print("  ╚═════════════════════════════════════════════════════════════╝")
    print()

    model = S3Pure(VIS_VOCAB, m_factors=args.m_factors).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Vocab:            {VIS_VOCAB} tokens — ↑ ↓ ← → EOS")
    print(f"  Geometry:         (S³)^{args.m_factors} = {4 * args.m_factors}D")
    print(f"  Parameters:       {n_params}  (embedding table only)")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Margin:           {args.margin}")
    print()

    # Data
    train_closed = generate_dataset(args.n_train)
    val_closed = generate_dataset(args.n_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("  Epoch │ Loss     │ σ closed │ σ open   │ Time")
    print("  ──────┼──────────┼──────────┼──────────┼──────")

    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        et0 = time.perf_counter()
        model.train()
        random.shuffle(train_closed)

        total_loss = 0.0
        total_sc = 0.0
        total_so = 0.0
        n_batches = 0

        for i in range(0, len(train_closed), args.batch_size):
            batch_closed = train_closed[i:i + args.batch_size]
            batch_open = [generate_open_walk() for _ in batch_closed]

            # Closed walks
            padded_c, lens_c = grid_pad_batch(batch_closed)
            tokens_c = torch.tensor(padded_c, dtype=torch.long, device=device)
            wlens_c = torch.tensor([walk_length(s) for s in batch_closed], dtype=torch.long)

            # Open walks
            padded_o, lens_o = grid_pad_batch(batch_open)
            tokens_o = torch.tensor(padded_o, dtype=torch.long, device=device)
            wlens_o = torch.tensor([walk_length(s) for s in batch_open], dtype=torch.long)

            sigma_c = model.sigma_at(tokens_c, (wlens_c - 1).clamp(min=0))
            sigma_o = model.sigma_at(tokens_o, (wlens_o - 1).clamp(min=0))

            loss = sigma_c.mean() + F.relu(args.margin - sigma_o).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sc += sigma_c.mean().item()
            total_so += sigma_o.mean().item()
            n_batches += 1

        elapsed = time.perf_counter() - et0
        print(
            f"  {epoch:>5} │ {total_loss / n_batches:>8.4f} │ "
            f"{total_sc / n_batches:>8.4f} │ {total_so / n_batches:>8.4f} │ {elapsed:>5.1f}s"
        )

    total_time = time.perf_counter() - t0
    print()
    print(f"  Training complete in {total_time:.1f}s.")
    print()

    # ── Inverse check ──
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Inverse discovery — spatial inverses?                     │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    model.eval()
    with torch.no_grad():
        e = model.get_embeddings()
        ud = F.normalize(qmul(e[UP], e[DOWN]), dim=-1)
        ud_sigma = torch.acos(torch.clamp(ud[:, 0].abs(), max=1 - 1e-7)).mean().item()

        lr = F.normalize(qmul(e[LEFT], e[RIGHT]), dim=-1)
        lr_sigma = torch.acos(torch.clamp(lr[:, 0].abs(), max=1 - 1e-7)).mean().item()

        eos_sigma = torch.acos(torch.clamp(e[GRID_EOS][:, 0].abs(), max=1 - 1e-7)).mean().item()

        print(f"  σ(UP · DOWN):     {ud_sigma:.6f}  (should be near 0)")
        print(f"  σ(LEFT · RIGHT):  {lr_sigma:.6f}  (should be near 0)")
        print(f"  σ(EOS):           {eos_sigma:.6f}  (should be near 0)")
        print()

        if ud_sigma < 0.1 and lr_sigma < 0.1:
            print("  Result:  PASS — model discovered both inverse pairs")
        elif ud_sigma < 0.3 or lr_sigma < 0.3:
            print("  Result:  PARTIAL — model partially discovered inverses")
        else:
            print("  Result:  FAIL — inverses not yet discovered")
        print()

    # ── σ separation ──
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  σ separation (closed vs open walks)                      │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    import statistics
    closed_sigmas = []
    open_sigmas = []

    with torch.no_grad():
        for seq in val_closed[:2000]:
            tokens = torch.tensor([seq], dtype=torch.long, device=device)
            wl = walk_length(seq)
            if wl > 0:
                _, sigmas = model.compose(tokens)
                closed_sigmas.append(sigmas[0, wl - 1].item())

        for _ in range(2000):
            seq = generate_open_walk()
            tokens = torch.tensor([seq], dtype=torch.long, device=device)
            wl = walk_length(seq)
            if wl > 0:
                _, sigmas = model.compose(tokens)
                open_sigmas.append(sigmas[0, wl - 1].item())

    n = min(len(closed_sigmas), len(open_sigmas))
    closed_sigmas = closed_sigmas[:n]
    open_sigmas = open_sigmas[:n]

    cl_mean = sum(closed_sigmas) / n
    op_mean = sum(open_sigmas) / n
    separation = op_mean - cl_mean

    cl_std = statistics.stdev(closed_sigmas) if n > 1 else 1e-12
    op_std = statistics.stdev(open_sigmas) if n > 1 else 1e-12
    pooled_se = math.sqrt(cl_std**2 / n + op_std**2 / n)
    t_stat = separation / pooled_se if pooled_se > 1e-12 else 0.0

    correct = sum(1 for c, o in zip(closed_sigmas, open_sigmas) if c < o)

    print(f"  σ (closed):        {cl_mean:.6f}  (mean over {n})")
    print(f"  σ (open):          {op_mean:.6f}")
    print(f"  Separation:        {separation:.6f}")
    print(f"  t-statistic:       {t_stat:.2f}")
    print(f"  Pair accuracy:     {correct}/{n} = {correct / n:.1%}")
    print()

    if t_stat > 2.576:
        print("  Result:  PASS — σ separates closed from open (p < 0.01)")
    elif t_stat > 1.96:
        print("  Result:  MARGINAL (p < 0.05)")
    else:
        print("  Result:  FAIL — no significant σ separation")
    print()

    # ── Generation ──
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Generation — geodesic nearest neighbor to C⁻¹            │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    with torch.no_grad():
        for temp in [0.0, 0.3, 0.5]:
            valid_t = 0
            total_t = 0
            examples = []
            for _ in range(500):
                start = random.choice([UP, DOWN, LEFT, RIGHT])
                tokens, sigmas = model.generate(start, 32, GRID_EOS, temperature=temp)
                closed = is_closed(tokens)
                total_t += 1
                if closed:
                    valid_t += 1
                if len(examples) < 4:
                    final_s = sigmas[-1] if sigmas else 0.0
                    s = walk_to_string(tokens)
                    examples.append((s, final_s, closed))

            pct = valid_t / total_t * 100
            label = "deterministic" if temp == 0.0 else f"temp={temp}"
            print(f"  {label:>15}:  {valid_t}/{total_t} = {pct:.1f}% closed")
            for s, sigma, closed in examples:
                tag = "closed" if closed else "open"
                print(f"    {s:<40} σ={sigma:.4f}  [{tag}]")
            print()


def main():
    parser = argparse.ArgumentParser(description="Drawing Board — Pure S³ geometry")
    parser.add_argument("task", choices=["brackets", "grid"], help="Task to train on")
    parser.add_argument("--m-factors", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task == "brackets":
        train_brackets(args)
    elif args.task == "grid":
        if args.m_factors < 2:
            print("  Grid walks need m >= 2 (2 DOF). Setting m=2.")
            args.m_factors = 2
        train_grid(args)


if __name__ == "__main__":
    main()