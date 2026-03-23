"""Brahman — learned composition on S³.

The model learns to map input tokens to unit quaternions on S³.
σ = arccos(|w|) measures coherence at every step, for free.

Step 1 (S3RNN): Sequential composition, σ-only feedback.
Step 2 (S3RNNValence): Hopf decomposition (σ, R, G, B, W) fed back.
Step 3 (S3Transformer): Parallel geodesic attention, same geometry.

EOS is NOT composed into the running product. It is a prediction target
only. The closure loss measures σ after the last bracket — the algebraic
content — not after structural tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EOS_TOKEN = 2


def qmul(q1, q2):
    """Hamilton product of two batches of quaternions.

    q1, q2: (..., 4) tensors — [w, x, y, z]
    Returns: (..., 4) tensor — the product.
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


class S3Embed(nn.Module):
    """Token → unit quaternion on S³."""

    def __init__(self, vocab_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, one_hot):
        return F.normalize(self.net(one_hot), dim=-1)


class S3Head(nn.Module):
    """Quaternion → logits over vocabulary."""

    def __init__(self, vocab_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, C):
        return self.net(C)


class S3RNN(nn.Module):
    """The S³ RNN. Step 1 of Brahman.

    Hidden state is a running quaternion product on S³.
    Loss = cross-entropy + λ · σ_final.

    EOS tokens are NOT composed. The running product only accumulates
    bracket tokens — the algebraic content. EOS is a prediction target
    only, so the closure loss measures the actual compositional state.
    """

    def __init__(self, vocab_size, hidden=64, closure_weight=0.1):
        super().__init__()
        self.embed = S3Embed(vocab_size, hidden)
        self.head = S3Head(vocab_size, hidden)
        self.closure_weight = closure_weight
        self.vocab_size = vocab_size

    def forward(self, tokens, targets=None, lengths=None):
        """
        tokens:  (B, T) token indices
        targets: (B, T) target indices (same as tokens for teacher forcing)
        lengths: (B,) number of bracket tokens per sequence (before EOS)
                 If None, uses all positions.
        """
        B, T = tokens.shape
        device = tokens.device
        C = torch.tensor([1., 0., 0., 0.], device=device).expand(B, 4).clone()

        all_logits = []
        all_sigma = []

        # Track per-sequence σ at the last bracket position
        bracket_sigma = torch.zeros(B, device=device)

        for t in range(T):
            tok = tokens[:, t]  # (B,)
            x = F.one_hot(tok, self.vocab_size).float()

            # Only compose non-EOS tokens into the running product
            g = self.embed(x)
            is_content = (tok != EOS_TOKEN).float().unsqueeze(-1)  # (B, 1)
            # If EOS/padding: compose identity (no change to C)
            identity = torch.tensor([1., 0., 0., 0.], device=device).expand(B, 4)
            g_masked = g * is_content + identity * (1.0 - is_content)
            C = F.normalize(qmul(C, g_masked), dim=-1)

            sigma = torch.acos(torch.clamp(C[:, 0].abs(), max=1 - 1e-7))
            all_logits.append(self.head(C))
            all_sigma.append(sigma)

            # Record σ at the last bracket position for each sequence
            if lengths is not None:
                at_last = (t == (lengths - 1)).float()
                bracket_sigma = bracket_sigma + sigma * at_last

        logits = torch.stack(all_logits, dim=1)
        sigmas = torch.stack(all_sigma, dim=1)

        if targets is not None:
            pred_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )

            # Closure loss: σ at the last bracket, not at the padded end
            if lengths is not None:
                closure_loss = bracket_sigma.mean()
            else:
                closure_loss = sigmas[:, -1].mean()

            total = pred_loss + self.closure_weight * closure_loss
            return logits, total, {
                "pred": pred_loss.item(),
                "closure": closure_loss.item(),
                "sigma_mean": sigmas.mean().item(),
                "sigma_final": bracket_sigma.mean().item() if lengths is not None else sigmas[:, -1].mean().item(),
            }
        return logits, sigmas

    @torch.no_grad()
    def generate(self, start_token, max_length, temperature=1.0):
        """Autoregressive generation. Returns (tokens, sigmas).

        σ is tracked only for bracket tokens. EOS is not composed.
        """
        device = next(self.parameters()).device
        tokens = [start_token]
        C = torch.tensor([[1., 0., 0., 0.]], device=device)
        sigmas = []

        for _ in range(max_length - 1):
            x = F.one_hot(
                torch.tensor([tokens[-1]], device=device), self.vocab_size
            ).float()
            g = self.embed(x)

            # Only compose bracket tokens
            if tokens[-1] != EOS_TOKEN:
                C = F.normalize(qmul(C, g), dim=-1)

            sigma = torch.acos(torch.clamp(C[:, 0].abs(), max=1 - 1e-7))
            sigmas.append(sigma.item())

            logits = self.head(C) / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

            if next_token == EOS_TOKEN:
                break

        return tokens, sigmas


# ── Step 2: Valence feedback ──────────────────────────────────


def hopf_valence(C):
    """Differentiable Hopf decomposition of running product.

    C: (B, 4) unit quaternion — [w, x, y, z]
    Returns: (B, 5) — [σ, R, G, B, W]

    σ = geodesic distance from identity
    R, G, B = Hopf projection to S² (base space — position channels)
    W = phase on S¹ (fiber — existence channel)
    """
    w, x, y, z = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    sigma = torch.acos(torch.clamp(w.abs(), max=1 - 1e-7))

    # Hopf base: project S³ → S² via (x, y, z) / ||(x, y, z)||
    denom = torch.clamp(1.0 - w * w, min=1e-12).sqrt()
    R = x / denom
    G = y / denom
    B = z / denom

    # Hopf fiber: phase angle
    W = torch.atan2((x * x + y * y + z * z).sqrt(), w)

    return torch.stack([sigma, R, G, B, W], dim=-1)


class S3EmbedWithValence(nn.Module):
    """Token + valence → unit quaternion on S³.

    Takes the one-hot token PLUS 5 valence channels as input.
    The model sees its own Hopf decomposition at every step.
    """

    def __init__(self, vocab_size, hidden=64):
        super().__init__()
        # +5 for valence: σ, R, G, B, W
        self.net = nn.Sequential(
            nn.Linear(vocab_size + 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, one_hot, valence):
        x = torch.cat([one_hot, valence], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class S3RNNValence(nn.Module):
    """The S³ RNN with valence feedback. Step 2 of Brahman.

    Same architecture as S3RNN, but the embedding receives the full
    Hopf decomposition (σ, R, G, B, W) of the running product at
    every step. The model doesn't just know "I'm drifting" — it knows
    which channels are breaking.
    """

    def __init__(self, vocab_size, hidden=64, closure_weight=0.1):
        super().__init__()
        self.embed = S3EmbedWithValence(vocab_size, hidden)
        self.head = S3Head(vocab_size, hidden)
        self.closure_weight = closure_weight
        self.vocab_size = vocab_size

    def forward(self, tokens, targets=None, lengths=None):
        B, T = tokens.shape
        device = tokens.device
        C = torch.tensor([1., 0., 0., 0.], device=device).expand(B, 4).clone()

        all_logits = []
        all_sigma = []
        bracket_sigma = torch.zeros(B, device=device)

        for t in range(T):
            tok = tokens[:, t]
            x = F.one_hot(tok, self.vocab_size).float()

            # Decompose current state into valence channels
            valence = hopf_valence(C)  # (B, 5)

            # Embed with valence feedback
            g = self.embed(x, valence)
            is_content = (tok != EOS_TOKEN).float().unsqueeze(-1)
            identity = torch.tensor([1., 0., 0., 0.], device=device).expand(B, 4)
            g_masked = g * is_content + identity * (1.0 - is_content)
            C = F.normalize(qmul(C, g_masked), dim=-1)

            sigma = torch.acos(torch.clamp(C[:, 0].abs(), max=1 - 1e-7))
            all_logits.append(self.head(C))
            all_sigma.append(sigma)

            if lengths is not None:
                at_last = (t == (lengths - 1)).float()
                bracket_sigma = bracket_sigma + sigma * at_last

        logits = torch.stack(all_logits, dim=1)
        sigmas = torch.stack(all_sigma, dim=1)

        if targets is not None:
            pred_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )
            if lengths is not None:
                closure_loss = bracket_sigma.mean()
            else:
                closure_loss = sigmas[:, -1].mean()

            total = pred_loss + self.closure_weight * closure_loss
            return logits, total, {
                "pred": pred_loss.item(),
                "closure": closure_loss.item(),
                "sigma_mean": sigmas.mean().item(),
                "sigma_final": bracket_sigma.mean().item() if lengths is not None else sigmas[:, -1].mean().item(),
            }
        return logits, sigmas

    @torch.no_grad()
    def generate(self, start_token, max_length, temperature=1.0):
        """Autoregressive generation with valence feedback."""
        device = next(self.parameters()).device
        tokens = [start_token]
        C = torch.tensor([[1., 0., 0., 0.]], device=device)
        sigmas = []

        for _ in range(max_length - 1):
            x = F.one_hot(
                torch.tensor([tokens[-1]], device=device), self.vocab_size
            ).float()

            valence = hopf_valence(C)
            g = self.embed(x, valence)

            if tokens[-1] != EOS_TOKEN:
                C = F.normalize(qmul(C, g), dim=-1)

            sigma = torch.acos(torch.clamp(C[:, 0].abs(), max=1 - 1e-7))
            sigmas.append(sigma.item())

            logits = self.head(C) / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

            if next_token == EOS_TOKEN:
                break

        return tokens, sigmas


# ── Step 3: S³ Transformer ──────────────────────────────────


class S3Attention(nn.Module):
    """Geodesic attention on S³.

    Attention scores = dot product of unit quaternions (cosine of
    geodesic distance). Small learned rotations per factor provide
    content-dependent routing — 4×4 per factor, not d×d.
    """

    def __init__(self, m_factors=1):
        super().__init__()
        self.m = m_factors
        self.q_rot = nn.Parameter(
            torch.eye(4).unsqueeze(0).repeat(m_factors, 1, 1)
        )

    def forward(self, x, mask=None):
        B, T, D = x.shape
        m = self.m

        x_q = x.view(B, T, m, 4)

        # Content-dependent query: rotate each factor
        q = torch.einsum('mij,btmj->btmi', self.q_rot, x_q)
        q = F.normalize(q, dim=-1)

        # Geodesic dot product per factor, averaged
        scores = torch.einsum('bimf,bjmf->bijm', q, x_q)
        scores = scores.mean(dim=-1)  # (B, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)

        # Aggregate in R^4, project back to S³
        agg = torch.einsum('bij,bjd->bid', weights, x)
        agg = agg.view(B, T, m, 4)
        agg = F.normalize(agg, dim=-1)
        return agg.view(B, T, D)


class S3Block(nn.Module):
    """Attention + quaternion composition residual.

    No FFN. No layer norm. The residual is quaternion multiplication
    (nonlinear, non-commutative). Normalization is the unit sphere
    constraint — one normalize call, zero parameters.
    """

    def __init__(self, m_factors=1):
        super().__init__()
        self.attention = S3Attention(m_factors)
        self.m = m_factors

    def forward(self, x, mask=None):
        B, T, D = x.shape
        m = self.m

        attended = self.attention(x, mask)

        # Residual via quaternion composition
        x_q = x.view(B, T, m, 4)
        a_q = attended.view(B, T, m, 4)
        out = F.normalize(qmul(x_q, a_q), dim=-1)
        return out.view(B, T, D)


class S3Transformer(nn.Module):
    """The S³ Transformer. Step 3 of Brahman.

    Same geometry as the RNN — unit quaternions on S³, closure loss,
    σ as coherence signal. But parallel: every position attends to
    every other via geodesic distance, not sequential composition.

    The architecture replaces:
      Q/K/V projections → geodesic dot product + tiny learned rotations
      FFN               → quaternion composition (the residual IS the FFN)
      Layer norm        → unit sphere constraint (free)
      Residual addition → quaternion multiplication (nonlinear, exact)
    """

    def __init__(self, vocab_size, m_factors=1, n_layers=4,
                 hidden=32, closure_weight=0.1, max_seq_len=64):
        super().__init__()
        self.m = m_factors
        self.dim = 4 * m_factors
        self.vocab_size = vocab_size
        self.closure_weight = closure_weight

        self.embed = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dim),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(max_seq_len, self.dim) * 0.02
        )
        self.blocks = nn.ModuleList([
            S3Block(m_factors) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, tokens, targets=None, lengths=None):
        B, T = tokens.shape
        device = tokens.device

        # Embed tokens to S³
        x = F.one_hot(tokens, self.vocab_size).float()
        x = self.embed(x)

        # Positional quaternions, normalize to sphere
        x = x + self.pos_embed[:T]
        x = x.view(B, T, self.m, 4)
        x = F.normalize(x, dim=-1)
        x = x.view(B, T, self.dim)

        # Causal mask — can't look ahead
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.head(x)

        # σ per position — mean across ALL S³ factors, not just the first
        x_q = x.view(B, T, self.m, 4)
        per_factor = torch.acos(torch.clamp(x_q[:, :, :, 0].abs(), max=1 - 1e-7))  # [B, T, m]
        sigmas = per_factor.mean(dim=-1)  # [B, T]

        # σ at last bracket position for closure loss
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0)
            bracket_sigma = sigmas[torch.arange(B, device=device), idx]
        else:
            bracket_sigma = sigmas[:, -1]

        if targets is not None:
            pred_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )
            closure_loss = bracket_sigma.mean()
            total = pred_loss + self.closure_weight * closure_loss
            return logits, total, {
                "pred": pred_loss.item(),
                "closure": closure_loss.item(),
                "sigma_mean": sigmas.mean().item(),
                "sigma_final": bracket_sigma.mean().item(),
            }
        return logits, sigmas

    @torch.no_grad()
    def generate(self, start_token, max_length, temperature=1.0, eos_token=None):
        """Autoregressive generation via full re-encoding."""
        if eos_token is None:
            eos_token = EOS_TOKEN
        device = next(self.parameters()).device
        tokens = [start_token]
        sigmas = []

        for _ in range(max_length - 1):
            tok_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            T = tok_tensor.shape[1]

            x = F.one_hot(tok_tensor, self.vocab_size).float()
            x = self.embed(x)
            x = x + self.pos_embed[:T]
            x = x.view(1, T, self.m, 4)
            x = F.normalize(x, dim=-1)
            x = x.view(1, T, self.dim)

            mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)
            for block in self.blocks:
                x = block(x, mask)

            # σ from last position — average across all m factors
            x_q = x[:, -1:].view(1, 1, self.m, 4)
            per_factor = torch.acos(torch.clamp(x_q[:, :, :, 0].abs(), max=1 - 1e-7))
            sigma = per_factor.mean().item()
            sigmas.append(sigma)

            logits = self.head(x[0, -1:]) / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

            if next_token == eos_token:
                break

        return tokens, sigmas
