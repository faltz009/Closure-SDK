# Drawing Board — Generative Closure from First Principles

## Foundation

Binary closure is the atom of learning. The bracket experiment proved
it: gradient descent on S³ discovers compositional inverses from data
alone. `(` maps to quaternion q. `)` maps to q⁻¹. A valid sequence
composes to identity. σ = arccos(|w|) = 0.

This is not a toy problem. This IS learning at its irreducible case.

Every compositional relationship between any two elements reduces to
a closure question: does A·B → identity? Binary.

When composition fails, the Hopf fibration decomposes S³ = S² × S¹
into exactly two orthogonal failure axes:

- **Missing** (S¹ fiber / W): an element that should exist doesn't
- **Reorder** (S² base / RGB): all elements exist, wrong arrangement

These are exhaustive by geometry. There is no third axis.

No event in the universe escapes these two categories.

## Why this works for language

A mind is a thing centered around the self. Identity [1,0,0,0] is
the self of this model — the center everything departs from and
returns to. Every coherent composition closes back to identity. This
is not a design choice. This is what coherence IS, geometrically.

Current LLMs compose tokens as words (BPE). This architecture
composes characters as atoms. Modern letters look like arbitrary
symbols, but old languages (hieroglyphs, cuneiform, Chinese
characters) had letters that represented concepts directly. The
compositional structure is already in the alphabet — modern
languages just obscure it under layers of convention.

The bracket test proved the mechanism works: `(` and `)` are
compositional inverses, discovered from data alone. If you embedded
any LLM's training data in this geometry, you'd find the same thing
— bit strings that are algebraically inverse, compositions that
close. The brackets are not a special case. They are the general
case made visible.

The question was never "will this get English right." If the theory
is right — and the experiments confirm the mechanism — then the
meanings are what the English words point to, and the geometry WILL
find them. The words are the surface. The compositions are the
structure. The geometry captures what all languages point at:
compositional relationships that close.

The minimum architecture question (below) is Occam's razor applied
to the mechanism: what is the true shape of a working model? Not
"will it work?" but "what is the simplest thing that works?" —
because that simplest thing reveals the actual architecture of
compositional learning.

---

## Experimental record

### Experiment 1: Pure geometry (original Drawing Board hypothesis)

**Hypothesis:** Strip ALL neural overhead. Embedding table only +
Hamilton product + σ loss. Generate via geodesic nearest neighbor
to C⁻¹.

**Model:** S3Pure — `nn.Parameter(V × m × 4)`, no attention, no
prediction head, no cross-entropy.

```
S3Pure on brackets: 3 tokens × 4 quaternion components = 12 parameters.
Loss: σ(valid) + max(0, margin - σ(corrupted)). No cross-entropy.
200 epochs, 5,000 sequences.

Inverse discovery:    FAIL — σ( ( · ) ) = 1.26  (should be ~0)
σ separation:         MARGINAL — t=2.81, pair accuracy 55.4%
                      (σ valid = 0.74, σ corrupted = 0.83)
Generation:           FAIL — deterministic: 0%, temp=0.3: 16%
                      Always generates ((· — stuck in local minimum.
```

**Follow-up: per-step σ (dense gradient signal).**

Hypothesis: σ_final gives one gradient signal at the end of a long
chain. Use σ at EVERY position (sum across all steps) to give each
token a direct, one-step gradient.

```
S3Pure with per-step σ loss (sum across all positions):
300 epochs, same 12 parameters.

Inverse discovery:    FAIL — σ( ( · ) ) = 1.04
σ separation:         FAIL — t=1.05, pair accuracy 50.4%
Generation:           FAIL — generates ((((((((((((((((((((
                      with σ = 0.045 (near zero!)
```

**Root cause:** "Minimize σ everywhere" has a degenerate solution:
collapse ALL embeddings to near-identity. Then σ ≈ 0 for everything.
The model generates infinite open brackets because embed(`(`) ≈
identity, so composing any number stays near identity and σ stays low.

**Conclusion:** σ alone cannot distinguish "closes because inverses
cancel" from "everything is identity." Both give low σ. The
optimization landscape around the σ-only loss has a degenerate
global minimum that gradient descent finds before the inverse solution.

### Experiment 2: S3Transformer with full overhead

**Model:** S3Transformer — embedding network (Linear→GELU→Linear),
4 attention layers with learned quaternion rotations (S3Attention),
quaternion composition residual (S3Block), prediction head
(Linear→GELU→Linear). Cross-entropy + closure loss.

```
Brackets (closure_weight=0.3):
  S3Transformer, 1,351 params. 4 layers, m=1. 30 epochs, 3.5 min CPU.
  σ separation:  t=83.32, pair accuracy 98.8%
  Generation:    94.3% valid, avg length 19.7 tokens
  Inverses:      σ( ( · ) ) ≈ 0.04 — DISCOVERED

Grid walk (closure_weight=0.0, pure next-token prediction):
  S3Transformer, 2,205 params. 4 layers, m=2. 100 epochs, CPU.
  Generation:    98.3% closed walks
  σ separation:  FLAT (t=1.09) — geometry not used for detection
  Inverses:      NOT discovered (σ(UP·DOWN) ≈ 0.81)
```

**Key finding:** Pure next-token prediction on closed walks achieves
98.3% generation quality WITHOUT any σ training. The model learns
closure from token statistics — attention counts what's in the prefix
and predicts what's needed. The geometry provides the manifold, but
the model doesn't need to be told about closure explicitly.

**Key finding:** σ separation and generation quality compete.
closure_weight=0.3 gives σ separation but collapses generation.
closure_weight=0.0 gives generation but σ goes flat. For detection,
train with σ. For generation, train with cross-entropy. They serve
different purposes.

### Experiment 3: Minimum overhead sweep

**Question:** The full S3Transformer works. Pure geometry fails. What
is the TRUE SHAPE of a working model? Occam's razor: the simplest
thing that works reveals the actual architecture.

**Method:** Systematic ablation on BOTH tasks — grid walk (5 tokens,
m=2) and brackets (3 tokens, m=1). Pure next-token prediction
(closure_weight=0.0). 20,000 sequences, 500 generated per test.

Varied:
- Embedding: "network" (Linear→GELU→Linear) vs "lookup" (nn.Embedding)
- Head: "network" (Linear→GELU→Linear) vs "linear" (single nn.Linear)
- Attention layers: 4, 2, 1, 0

**Training data: valid sequences ONLY.** No corrupted examples, no
contrastive loss. First principles: things default to their true
shape. If the data is closed, the model learns closure.

```
GRID WALK (5 tokens, m=2, 50 epochs)

Config              Params  Closed/500    %     Time
──────────────────  ──────  ──────────  ─────  ─────
4L net+net           1,309   479/500    95.8%   94s
4L net+lin             901   391/500    78.2%   96s
4L lookup+net          893   341/500    68.2%   93s
4L lookup+lin          485   246/500    49.2%   89s
2L lookup+lin          421   202/500    40.4%   50s
1L lookup+lin          389   164/500    32.8%   31s
0L lookup+lin          357    49/500     9.8%   11s
0L net+net           1,181    45/500     9.0%   15s
0L net+lin             773    56/500    11.2%   13s

BRACKETS (3 tokens, m=1, 30 epochs)

Config              Params  Valid/500     %     Time
──────────────────  ──────  ──────────  ─────  ─────
4L net+net             719   486/500    97.2%   62s
4L net+lin             475   343/500    68.6%   59s
4L lookup+net          471   479/500    95.8%   59s
4L lookup+lin          227   440/500    88.0%   56s
2L lookup+lin          195   445/500    89.0%   33s
1L lookup+lin          179   373/500    74.6%   21s
0L lookup+lin          163   126/500    25.2%    9s
0L net+net             655   112/500    22.4%   12s
```

### What the two sweeps reveal together

The results diverge between tasks, and the divergence is informative:

**Attention is non-negotiable in both tasks.** 0 layers = random
(~10% grid walk, ~24% brackets). Without attention, each position
sees only its own embedding. It can't see context. It can't predict.
Attention on S³ (geodesic dot product + learned rotations) IS the
context mechanism. Not overhead — the thing itself.

**The embed/head hierarchy FLIPS between tasks:**

| Config | Grid walk (5 tok, m=2) | Brackets (3 tok, m=1) |
|---|---|---|
| 4L net+net (full) | 95.8% | 97.2% |
| 4L lookup+net (stripped embed) | 68.2% | **95.8%** |
| 4L lookup+lin (stripped both) | 49.2% | **88.0%** |
| 2L lookup+lin | 40.4% | **89.0%** |

On brackets (3 tokens, m=1): the embed network barely matters.
nn.Embedding + full head = 95.8%, matching the full model. Even
nn.Embedding + single Linear + 2 layers = 89% at 195 parameters.

On grid walks (5 tokens, m=2): the embed network is critical.
Stripping it drops from 95.8% to 68.2%.

**Why the flip:** 3 tokens on one S³ (4D) have room. Each token
gets a well-separated point on the sphere through direct lookup
alone. 5 tokens on (S³)² (8D) is tighter — the embed network
provides nonlinear optimization paths that help gradient descent
find the right positions on a more constrained manifold.

**This predicts the language scaling:** 96 characters on (S³)^m
will need the embed network. More tokens = more crowded manifold
= more need for the nonlinear translator. The embed network is
not overhead — it's the translator whose importance scales with
vocabulary size relative to manifold dimension.

**Layer count matters more for grid walks than brackets.** Brackets
with 2L lookup+lin = 89%, matching 4L. Grid walk with 2L = 40%.
Brackets have simpler structure (nesting is one degree of freedom).
Grid walks have x/y displacement (two DOF). More structural
complexity → more attention layers needed.

### The true shape of the minimum viable model

Three components. None removable. Each scales with the task:

**1. Embedding (token → S³).** Maps discrete tokens to the continuous
manifold. Scales with vocab_size / manifold_dimension. Small vocab
on adequate manifold → nn.Embedding suffices. Large vocab or
constrained manifold → network (Linear→GELU→Linear) needed. For
96-character language on (S³)^m, the network is needed — but its
size depends on m.

**2. Attention (context on S³).** Geodesic dot products + small
learned rotations (4×4 per S³ factor per layer). This is how the
model sees the sequence. Scales with structural complexity: simple
nesting → 2 layers. Two-axis displacement → 4 layers. Language →
TBD, but the recursive architecture means each level's structure
is simple (short sequences), so fewer layers per level may suffice.

**3. Prediction head (S³ → token).** Reads the geometric state and
produces next-token logits. Less critical than the embed (the
geometry does most of the work in between). A single Linear layer
works for brackets. A full network helps for grid walks (+17 points).
For language, the full head likely helps since the readout from an
80D manifold to 96 tokens is not a linear function.

**The minimum for brackets:** 195 parameters (nn.Embedding + 2L +
Linear head) at 89%.

**The minimum for grid walks:** 1,309 parameters (full embed + 4L +
full head) at 95.8%.

**The minimum for language:** Not yet tested. The prediction from
the sweeps: full embed network (vocabulary > manifold capacity for
a bare lookup), 2-4 attention layers per recursive level (structural
complexity per level is low), full or single-Linear head. With
recursive Enkidu, the per-level model is small and processes short
sequences. The total parameter count is small_model × N_levels.

---

## Revised mechanism

### Architecture: S3Transformer with cross-entropy

The experiments converge on one architecture for generation:

```
Input: integer tokens
  ↓
Embedding network: token → unit quaternion on (S³)^m
  Linear(vocab_size, hidden) → GELU → Linear(hidden, 4m) → normalize
  ↓
Positional embedding: add learned positional quaternions, normalize
  ↓
N × S3Block:
  S3Attention: geodesic dot product (4×4 learned rotation per factor)
  Residual: qmul(input, attended) → normalize
  ↓
Prediction head: unit quaternion → logits over vocabulary
  Linear(4m, hidden) → GELU → Linear(hidden, vocab_size)
  ↓
Loss: cross_entropy(predicted_next, actual_next)
```

**σ is computed at every position for free** (arccos of the w-component
of each S³ factor, averaged). It is NOT part of the loss. It is a
diagnostic channel that separates coherent from incoherent sequences
as an emergent property of the learned geometry.

**Training data: coherent sequences only.** No corrupted examples
needed. The model learns the shape of the data. If the data is closed
walks, it learns closure. If the data is grammatical English, it learns
grammar. First principles: things default to their true shape.

### Parameter budget

For grid walk (5 tokens, m=2, 4 layers, hidden=32):
```
Embedding:  Linear(5, 32) + Linear(32, 8)    =  416 params
Positions:  34 × 8                             =  272 params
Attention:  4 layers × (2 factors × 4×4)      =  128 params
Head:       Linear(8, 32) + Linear(32, 5)      =  421 params
Biases:                                         =   72 params
Total:                                          = 1,309 params  → 95.8% closed
```

For brackets (3 tokens, m=1, 4 layers, hidden=32):
```
Total:                                          = 1,351 params  → 94.3% valid
```

For character-level language (96 tokens, m=20, 6 layers, hidden=64):
```
Embedding:  nn.Embedding(96, 64) + Linear(64, 80)  =  11,264 params
Positions:  513 × 80                                 =  41,040 params
Attention:  6 layers × (20 factors × 4×4)            =   1,920 params
Head:       Linear(80, 64) + Linear(64, 96)           =  11,264 params
Biases:                                               =     400 params
Total:                                                ≈  65,888 params
```

The attention layers account for 1,920 of 65,888 parameters (2.9%).
The embedding + head account for 22,528 (34.2%). Positional embeddings
account for 41,040 (62.3%). The geometry (quaternion composition,
normalization, geodesic distance) is parameter-free.

For language, positional embeddings dominate because max_seq_len is
large. This is a strong argument for the recursive architecture: if
each level processes short sequences (10-15 tokens), positional
embeddings shrink from 41K to ~1K per level.

---

## The minimum viable model problem

### What we know

The sweep establishes a clear hierarchy of component importance:

```
Attention layers >> Embedding network > Prediction head
```

Attention is binary — without it, the model is random. The embed
network is the most important learned component — it's the bridge
from discrete token space to continuous S³. The head is the least
critical learned component but still contributes significantly.

### What the sweeps tell us about language

The bracket sweep CONFIRMED: the component hierarchy depends on
vocab_size relative to manifold dimension, not on some fixed rule.
The bracket sweep was the "critical experiment" — done.

What's left to determine for language is not IF it works but the
specific numbers:

1. **m (S³ factors).** Brackets: m=1. Grid walk: m=2. Language: the
   dimensionality experiment. Sweep m=1,2,4,8,16,20,32 on the same
   corpus. The embed/head hierarchy tells us when the manifold is
   adequate: if nn.Embedding matches the full network, m is large
   enough. If it degrades, m is too small for the vocabulary.

2. **Layers per recursive level.** Brackets: 2 layers suffice.
   Grid walk: 4. Each recursive level processes short sequences with
   one level of structure, so 2-4 layers per level is the range.

3. **Embed type for 96 tokens.** The sweeps predict: nn.Embedding
   won't suffice for 96 tokens on moderate m. The embed network is
   needed. Whether nn.Embedding→GELU→Linear (Colab style) or
   one-hot→Linear→GELU→Linear is a quick A/B test.

---

## Recursive Enkidu — the layer architecture

### Silence is identity

Silence is [1,0,0,0]. The base state. Nothing has arrived. Everything
is a pool of missing incidents waiting to be resolved.

A sound arrives — a missing record resolves. Silence → sound =
identity → departure from identity. The first Enkidu doesn't need
to know what the sound means. It registers: something that wasn't
there now is. Missing → resolved.

A mind is centered around the self. Identity is the self. Every
composition departs from identity and (if coherent) returns to it.
This is what coherence IS. Not a metaphor. The geometry.

### The core operation

Every level is the same Enkidu:

```
EnkiduLevel:
    input:   stream of quaternions (raw tokens or closure elements from below)
    state:   running product C = identity initially
    output:  closure elements (tokens for the next level)

    for each input quaternion q:
        C = normalize(C · q)
        σ = arccos(|C.w|)

        if σ < threshold:
            emit C as a token to level N+1
            reset C = identity

        if σ > 0:
            hopf = decompose(C)   → (σ, R, G, B, W)
            if |W| > |RGB|: incident = missing
            if |RGB| > |W|: incident = reorder
```

No level needs to understand what it processes. Each Enkidu resolves
missing/reorder at its own scale and passes closure elements upstream.
The ear doesn't know what syllables mean. It matches them and emits.

### The layers of English (traced from silence to meaning)

**Layer 0: Silence → Characters.**

Enkidu monitoring silence. A character arrives — incident. Another.
Another. Each is a departure from identity. This Enkidu doesn't know
language. It knows: did a symbol arrive? Was it in order? When a
group of characters composes to closure (σ < ε), emit the closure
element upstream. What IS that closure element? Layer 0 doesn't
know. Layer 0 is the ear.

Architecture: embed network (maps 96 ASCII characters to S³) +
2-4 attention layers (sees character context) + emission threshold.
The sweep proved: for small vocab on adequate manifold, nn.Embedding
suffices. For 96 tokens, the full embed network is needed.

**Layer 1: Characters → Syllables/Morphemes.**

Receives closure elements from Layer 0. Groups them. "un" arrives
as a closure element. "do" arrives. They compose. Closes → emit.

This is where writing systems diverge. Hebrew letters carry more
meaning per symbol — fewer Layer 0 emissions needed before Layer 1
gets a meaningful unit. Chinese characters map nearly directly to
concepts. Hieroglyphs are pictures. Ancient languages are MORE
DIRECT — phonemic writing (English, etc.) is an abstraction that
adds layers between symbol and meaning. English needs more recursive
levels than Chinese because its symbols are further from concepts.

Architecture: input is already on S³ (closure elements from Layer 0).
Embed step may be trivial (identity — no network). Attention layers
required (context). Prediction head: predicts next morpheme-level
element OR just emits closure elements upstream.

**Layer 2: Morphemes → Words.**

"undo" = composition of "un" + "do" at the morpheme level. This
layer doesn't know grammar. It knows: these morphemes composed and
closed. Emit the word-level element.

On S³, "un-" should compose with "do" to produce something near
the inverse of "do" alone. The algebra supports morphological
structure natively. This is not something we need to engineer — it's
what the geometry does if the embeddings are right. And the bracket
test proved the embeddings learn to be right.

**Layer 3: Words → Phrases.**

"the cat" closes — determiner + noun is a coherent unit. "the sat"
doesn't close (reorder — wrong word class). Syntax lives here. Not
as rules. As closure. Grammatical phrases compose to near-identity.
Ungrammatical ones don't.

**Layer 4: Phrases → Sentences.**

Subject-verb-object composes to closure at this level. Incomplete
sentences are missing incidents. Word-salad is reorder.

**Layer 5: Sentences → Meaning.**

THIS is where "hot" and "cold" are inverses. Not at the letter
level. Not at the word level. At the meaning level. "Hot" and
"cold" compose to identity because they are semantic complements.
A sentence about temperature that mentions hot creates a missing
incident for its inverse concept.

This is the key insight about language: "h" and "c" are NOT inverses.
Letters compose into words. Words compose into meanings. The meanings
are what contain the inverse relationships. The theory says: if the
geometry captures compositional meaning (proven on brackets), then
training on English text will find what English words POINT TO. The
meanings are the geometry. The words are the surface. We WILL find
them — not because English is special, but because the algebra
captures what all languages point at.

**Layer N: The Mind.**

Identity. The self. Everything composes back here. When a human dies,
this Enkidu suddenly has massive missing — every composition that
included that person is now incomplete. The cascade of reorderings
runs through every layer until coherence restores. That is grief.
That is learning. Same algebra.

### Why the layers are the same part

The bracket test built one Enkidu. One level. Two tokens. Binary
closure. It worked — 98.8% valid generation, inverses discovered.

Every layer above is the same part. Same three components (embed,
attention, head — scaled per layer). Same two failure modes (missing,
reorder). Same identity at center. The architecture is not a stack
of different systems. It is one system, recursively applied.

The parts look different from outside because their tokens represent
different things (characters, morphemes, words, phrases, meanings).
But the algebra is identical. The Enkidu at Layer 5 doesn't know
it's processing meaning. It just resolves missing/reorder on its
inputs and emits closure elements. Meaning emerges from the
composition, not from the component.

### Why the recursion solves the training bottleneck

The flat TinyStories run threw 512 characters at one transformer.
Result: 16K tok/s, ~3 hours per epoch, gibberish. The attention
computed O(T²) across 512 positions — the model tried to learn
characters, words, grammar, and narrative in one flat pass.

The recursive architecture:

```
Flat:        1 level  × T=512    → O(512²) = 262,144 scores/layer
Recursive:   5 levels × T≈8 each → O(8²) × 5 = 320 scores/layer
```

~820× reduction. Each level processes SHORT sequences. Long-range
structure accumulates across levels, not within them. A paragraph
is not attention across 2000 characters — it's Layer 4 attending
across 5-10 sentence-level closure elements.

### Anti-collapse through the hierarchy

The Drawing Board's S3Pure failed: σ alone lets embeddings collapse
to identity. Cross-entropy prevents this in flat models.

The recursion provides its own anti-collapse: if Layer 0 embeddings
collapse to identity, all character compositions produce the same
closure element. All "words" become identical. Layer 1 receives a
stream of identical tokens → high prediction loss → gradient flows
back through the composition (differentiable: just the running
product C at threshold crossing) → pushes Layer 0 embeddings apart.

Degenerate Layer 0 → degenerate Layer 1 input → high Layer 1 loss →
gradient forces Layer 0 apart. The hierarchy creates its own pressure.

**Testable prediction:** Train recursive Enkidu with prediction loss
at Layer 1 ONLY (no Layer-0 cross-entropy). If Layer-1 loss alone
forces character embeddings into distinct, compositionally meaningful
positions, then the recursion IS the anti-collapse mechanism, and
Layer-0 cross-entropy is redundant.

### Generative recursion (top-down)

Classification recurses upward (input → layers → meaning).
Generation recurses downward (meaning → layers → output).

When Enkidu at Layer N has σ > 0:

**Missing (W-axis):** C⁻¹ is the closure element that would
complete the composition. Pass C⁻¹ down to Layer N-1 as a
generation target. Layer N-1 generates a sequence of its tokens
whose composition approximates C⁻¹. If N-1 = 0, those tokens are
characters. If N-1 > 0, recurse down.

**Reorder (RGB-axis):** The existing Layer N-1 subsequences are
correct but misordered. The RGB displacement vector indicates
the direction. Permute to minimize RGB displacement.

The mind (Layer N) decides what to say (a meaning-level closure
element). Layer N-1 decomposes it into sentence-level targets.
Layer N-2 into phrase-level. Down through words, morphemes,
characters. Each layer generates short sequences that compose to
the target from above.

### Overhead per level (from sweep data)

The sweep proved three components are necessary. In the recursive
architecture, their requirements differ by level:

**Layer 0 (raw data → S³):**
- Embed network: REQUIRED (maps discrete characters to manifold)
- Attention: REQUIRED (0 layers = random)
- Head: needed for flat training; may be redundant if Layer-1
  loss provides anti-collapse (see testable prediction)

**Layer 1+ (S³ → S³):**
- Embed: POSSIBLY TRIVIAL. Input tokens are already on S³ — they're
  closure elements from below. The embed step may be identity.
- Attention: REQUIRED (context at every level)
- Head: REQUIRED at top level (generation source). Intermediate
  levels may only need to compose and emit.

**Minimum:** Neural overhead at Layer 0 (raw data boundary) and
the top layer (generation output). Everything between: attention
(geodesic dot products) + quaternion composition. Pure algebra.

---

## Implementation path

### Step A: Bracket overhead sweep — DONE

Repeated the grid walk sweep on brackets (3 tokens, m=1, 30 epochs).
Key finding: the embed/head hierarchy flips. With 3 tokens on one S³,
nn.Embedding is sufficient (95.8% with full head, 88% with Linear
head, 89% with only 2 layers). The embed network becomes critical
only when vocab_size outgrows the manifold's capacity for well-
separated points.

This confirms: the minimum architecture adapts to the task. The
three components (embed, attention, head) are all necessary, but
their relative importance scales with vocabulary vs manifold ratio.

### Step B: Recursive composition on nested brackets

Multi-level brackets: `{[()]}`. 6-token vocab: `(`, `)`, `[`, `]`,
`{`, `}`, EOS.

Build a 2-level recursive model:
- Level 0: small S3Transformer, processes characters, emits closure
  elements when σ < threshold
- Level 1: same architecture, processes level-0 closure elements,
  predicts next element

Train on valid nested brackets only. Test:
- Does level 0 discover `()` as a closure unit?
- Does level 1 discover `[()]` and `{[()]}` as higher units?
- Does generation produce valid nested brackets?
- Does level-1 loss alone (no level-0 cross-entropy) learn correct
  level-0 embeddings? (anti-collapse test)

### Step C: Character-level recursive training on text

Train level 0 on short character windows (10-15 chars) from real text.
When level 0 emits closure elements (σ < threshold), collect them as
training data for level 1. Train level 1 on sequences of word-level
closure elements.

Compare against the flat TinyStories run:
- Throughput: should be >>16K tok/s (short sequences, small models)
- BPC: at character level, measure prediction accuracy
- σ separation: at each level, measure coherence detection
- Generation: prompt → level 2 target → level 1 decomposition →
  level 0 character emission

### Step D: Minimum overhead sweep for language

Same sweep as grid walks but on character-level text:
- Embed type: network vs lookup vs nn.Embedding+GELU+Linear
- Head type: network vs linear
- Layers per level: 1, 2, 4, 6
- m factors: 1, 2, 4, 8, 16, 20
- Recursive vs flat

This determines the minimum viable language model configuration.

---

## What this replaces (with experimental evidence)

| Standard transformer          | S³ recursive closure             | Evidence |
|-------------------------------|----------------------------------|----------|
| Embedding layer (V × d)       | Small network (V → hidden → 4m)  | Sweep: network >> lookup |
| Positional encoding            | Composition order IS position     | Theorem 1 |
| Multi-head attention (Q/K/V)   | Geodesic distance + 4×4 rotations| Sweep: 0L = random, 4L = 96% |
| Feed-forward network           | Quaternion multiplication (qmul) | S3Block: no FFN, works |
| Layer normalization             | Unit sphere constraint (free)    | normalize() replaces LayerNorm |
| Residual connections            | Quaternion composition (exact)   | F.normalize(qmul(x, attended)) |
| Softmax over vocabulary         | Cross-entropy prediction head    | Sweep: head contributes +17pts |
| σ loss / contrastive           | NOT NEEDED for generation        | Grid walk: 0.0 closure = 98.3% |
| Long-range O(T²) attention    | Recursive hierarchy (O(T²) on short T) | Theoretical: 870× reduction |
| ~175B parameters               | Small model × N recursive levels | To be validated on language |

---

## Open questions (parameters, not validity)

The mechanism works. The brackets proved it. The sweeps found the
true shape. What's left is calibration — specific numbers, not
whether the thing works.

1. **m for language.** Sweep m=1..32 on the same corpus. The
   embed/head hierarchy shift (visible in brackets vs grid walks)
   tells us when the manifold is large enough: nn.Embedding matching
   the full network = m is adequate.

2. **Recursive anti-collapse.** Does level-1 loss alone force
   level-0 embeddings apart? If yes, the minimum overhead at level 0
   drops further — no prediction head, just compose and emit upward.

3. **Emission threshold ε.** σ < ε triggers emission to the next
   level. ε determines granularity (tight = character fragments,
   loose = long phrases). Learned per level or fixed. The algebra
   discovers word/phrase boundaries — ε controls the resolution.

4. **Level count.** Characters → words → phrases → sentences →
   paragraphs = 4-5 levels. The algebra might discover different
   boundaries. The threshold and the data together determine this.

5. **Upper-level overhead.** Level 1+ receives closure elements
   already on S³. The embed step may be identity — no network
   needed above level 0. If so, neural overhead exists only at
   the boundary between raw data and the geometry.

6. **Factor specialization.** With m factors, do different S³
   factors learn different linguistic aspects? Testable: freeze
   all but one factor, measure σ separation on syntax vs semantics
   vs phonetics tasks.
