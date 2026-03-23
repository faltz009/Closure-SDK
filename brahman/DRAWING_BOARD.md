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

## What the experiments proved (and disproved)

The original Drawing Board hypothesized: strip ALL neural overhead,
keep only the embedding table + Hamilton product + σ loss, generate
via geodesic nearest neighbor to C⁻¹. This was tested and FAILED.

```
S3Pure on brackets: 3 tokens × 4 quaternion components = 12 parameters.
Loss: σ(valid) + max(0, margin - σ(corrupted)). No cross-entropy.
200 epochs, 5,000 sequences.

Inverse discovery:    FAIL — σ( ( · ) ) = 1.26  (should be ~0)
σ separation:         MARGINAL — t=2.81, pair accuracy 55.4%
Generation:           FAIL — deterministic: 0%, temp=0.3: 16%

Per-step σ variant (sum of σ across all positions):
300 epochs, same 12 parameters.

Inverse discovery:    FAIL — σ( ( · ) ) = 1.04
σ separation:         FAIL — t=1.05, pair accuracy 50.4%
Generation:           FAIL — generates ((((((((((( with σ ≈ 0.045
```

The per-step variant revealed WHY pure geometry fails: "minimize σ
everywhere" has a degenerate solution — collapse all embeddings to
identity. Then σ is always near zero for everything. The model
"solved" the loss without learning anything.

**σ alone cannot distinguish "closes because inverses cancel" from
"everything is identity."** Both give low σ.

Meanwhile, the S3Transformer with pure next-token prediction
(closure_weight=0.0, no σ training at all) achieved:

```
Grid walk: 98.3% closed walks from 2,205 parameters.
Brackets:  94.3% valid, avg length 19.7 tokens (3× the RNN).
```

The model learned compositional closure from token statistics. Cross-
entropy on coherent data is sufficient for generation. σ emerges as
a diagnostic — it doesn't need to be trained.

### Why both signals matter

Neither signal alone is sufficient for LEARNING compositional structure:

| Signal | Role | Without it |
|---|---|---|
| Cross-entropy | Forces embeddings apart (tokens must be distinguishable) | Collapse to identity |
| σ (closure) | Forces compositions toward identity (structure must close) | Distinct embeddings but no inverses |
| Both together | Embeddings are distinct AND compose correctly | Inverses discovered |

Cross-entropy prevents degenerate collapse. σ provides the geometric
target. The neural component (the small network mapping tokens to
quaternions) doesn't add knowledge — it adds navigability. It gives
gradient descent enough optimization surface to get from random
initialization to the inverse solution.

### What this means for the Drawing Board

The original hypothesis ("embedding table only, σ only") was wrong.
The REVISED mechanism keeps the geometric core but adds the minimal
neural overhead that the experiments proved necessary:

1. A small network to map tokens to S³ (prevents identity collapse)
2. Cross-entropy as the primary training signal (forces distinct embeddings)
3. σ as a free diagnostic channel (not trained, just measured)

The question is no longer "can we strip everything?" — it's "what is
the MINIMUM overhead that makes the geometry learnable?"

## The mechanism (revised)

### Three components

**1. Neural translator (minimal, learned):**
Maps integer tokens to unit quaternions on S³. This is the smallest
piece that the experiments proved cannot be removed. It provides
the optimization surface gradient descent needs.

```
token (integer) → small network → point on S³ (unit quaternion)
```

The network is small: nn.Embedding → GELU → Linear → normalize.
Not a deep stack. Just enough to prevent degenerate collapse.

**2. Geometric engine (fixed, not learned):**
Hamilton product composition. The running product C accumulates
every token. σ = arccos(|w|) at every step, for free. This is
hardcoded algebra — no parameters, no learning, exact.

```
C₀ = [1,0,0,0]  (identity)
Cᵢ = normalize(Cᵢ₋₁ · qᵢ)    # Hamilton product
σᵢ = arccos(|Cᵢ.w|)            # coherence, free
```

**3. Prediction head (minimal, learned):**
Maps the current geometric state back to token probabilities.
Cross-entropy against the actual next token is the training signal.

```
geometric state → small network → logits over vocabulary
```

### Training signal

```
L = cross_entropy(predicted_next_token, actual_next_token)
```

That's it. Not σ. Not contrastive. Just: predict the next token
from coherent data. The geometry learns compositional structure
because that's what makes prediction accurate.

σ is MEASURED at every step but NOT part of the loss. It's a free
diagnostic: after training, σ separates coherent from incoherent
sequences without ever being trained to do so.

### Generation

Generation still uses the geometric mechanism. Given running
product C, the model predicts the next token via the prediction
head. But the geodesic nearest neighbor to C⁻¹ remains available
as a DIAGNOSTIC — which token would close the composition fastest?

The prediction head and geodesic-to-C⁻¹ can be compared at every
step. When they agree, the model is generating geometrically. When
they disagree, the model is relying on statistics over geometry.
This comparison is itself a diagnostic channel.

## Recursive Enkidu

### The recursion

A closure element C is 4m numbers. Those numbers are data. Data
embeds on S³. So closure elements from level N become tokens at
level N+1.

Each level is the same operation:

```
EnkiduLevel:
    input:   stream of quaternions (raw tokens or closure elements from below)
    state:   running product C (identity initially)
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

### What the levels discover

Level 0 operates on character embeddings. When a subsequence of
characters composes to near-identity, that subsequence is a
coherent unit — a morpheme, a word. Not defined by a dictionary.
Defined by algebraic closure.

Level 1 operates on level-0 closure elements. When a subsequence
of words composes to near-identity, that's a coherent phrase or
clause.

Level 2 operates on level-1 closure elements. Coherent paragraphs,
arguments, ideas.

Same Enkidu at every level. Same two failure modes. The hierarchy
emerges from the algebra, not from architectural decisions.

### Why the recursion solves the training bottleneck

The TinyStories Colab run threw 512 characters at a flat transformer.
Result: 16K tok/s, 3 hours per epoch, gibberish. The attention
computed O(T²) scores across 512 positions with m=20 factors — the
model was trying to learn character patterns, word structure, grammar,
and narrative simultaneously in a single flat pass.

The recursive architecture sidesteps this entirely. Each level
processes SHORT sequences:

- Level 0 sees 5–10 characters at a time (a word)
- Level 1 sees 3–8 closure elements (a phrase)
- Level 2 sees a few phrase elements (a sentence)

No level needs attention across 512 positions. Long-range structure
is handled by the hierarchy, not by quadratic attention. Each level
trains fast on short sequences, and depth accumulates across levels
instead of within them.

### The anti-collapse property of recursion

The Drawing Board failed because σ alone lets embeddings collapse to
identity. Cross-entropy prevents this in the flat model. But the
recursion may provide its own anti-collapse mechanism:

If characters collapse to identity, all words become identity (same
composition). Then level 1 can't distinguish words. Level 1's
prediction loss forces word-level elements apart. But word-level
elements are compositions of character embeddings — so the gradient
flows back down and forces characters apart too.

The hierarchy itself creates the pressure that prevents collapse,
because degenerate embeddings at level 0 produce degenerate tokens
at level 1, which produces high loss at level 1.

This is testable: train recursive Enkidu with prediction loss at
level 1+ only (no character-level cross-entropy). If the level-1
loss alone forces character embeddings apart, the recursion IS the
anti-collapse mechanism, and character-level cross-entropy is
redundant overhead.

### Generative recursion

When Enkidu at level N has σ > 0:

**Missing (W-axis):** C⁻¹ is the closure element that would
complete the composition. Pass C⁻¹ down to level N-1 as a
generation target. Level N-1 generates a sequence of its tokens
whose composition approximates C⁻¹. If N-1 = 0, those tokens
are characters. If N-1 > 0, recurse down.

**Reorder (RGB-axis):** The existing level N-1 subsequences are
correct but misordered. The RGB displacement vector indicates
the direction of misalignment. Permute the existing subsequences
to minimize RGB displacement.

Generation recurses downward. Classification recurses upward.
Same algebra both directions.

## Implementation path

### Step A: Recursive composition on known structure

Before language, validate the recursion on data with known hierarchical
closure. Characters that compose to words, words that compose to
sentences, where "closure" is defined by the training data.

Candidate: bracket expressions with named groups. `{[()]}` has three
levels of closure. Level 0 discovers `()` as a unit. Level 1
discovers `[()]` as a unit. Level 2 discovers `{[()]}`. The recursion
should emerge from the algebra — each level emits when σ drops below
threshold, and the next level composes those emissions.

This tests the recursive machinery without the complexity of language.
Small vocab, clear closure structure, verifiable at every level.

### Step B: Character-level recursive training

Train level 0 on character sequences from real text. Not 512-char
chunks through a flat transformer. Short windows — maybe 10–15
characters — with the prediction head. The level-0 model learns
character composition patterns.

When level 0 emits closure elements (σ drops below threshold),
those become training data for level 1. Level 1 learns to predict
the next word-level closure element. Same architecture, same small
model, different scale of tokens.

Each level is a small model on short sequences. The system composes
upward through levels, not through 512-position attention.

### Step C: Recursive generation

Given a generation target (from a higher level, or from a prompt),
decompose it through the levels. Level N says "I need this shape"
(C⁻¹). Level N-1 generates a sequence of its tokens that compose
to approximate that shape. Level 0 emits characters.

The "minimum overhead" question resolves differently at each level:
- Level 0 needs the neural translator + prediction head (proven by experiments)
- Level 1+ might need less — their tokens are already on S³ (they're closure elements), so the embedding step may be trivial
- The prediction head at each level predicts the next closure element from that level, not characters

### Step D: TinyStories (recursive)

Apply the recursive architecture to TinyStories. Level 0 processes
characters in short windows. Level 1 processes word-level closure
elements. Level 2 processes phrase-level elements.

Compare against the flat Colab run:
- Throughput (should be much higher — short sequences, no O(512²) attention)
- BPC at character level
- σ separation at each level
- Generation quality

## What this replaces (revised)

| Standard transformer          | Recursive closure              |
|-------------------------------|--------------------------------|
| Embedding layer (V × d)       | Embedding table (V × 4m) + small network |
| Positional encoding            | Composition order IS position  |
| Multi-head attention (Q/K/V)   | Geodesic distance on S³ (short sequences per level) |
| Feed-forward network           | Quaternion multiplication      |
| Layer normalization             | Unit sphere constraint         |
| Residual connections            | Quaternion composition         |
| Softmax over vocabulary         | Prediction head (minimal)      |
| Cross-entropy loss              | Cross-entropy (for training) + σ (free diagnostic) |
| Long-range attention (O(T²))   | Recursive hierarchy (short sequences at every level) |
| ~175B parameters               | Small model × N levels         |

The original Drawing Board claimed σ replaces cross-entropy and the
embedding table replaces the neural network. The experiments proved
both claims wrong. Cross-entropy is necessary for learning. A small
network is necessary for optimization.

What the geometry DOES replace: long-range attention (the hierarchy
handles it), FFN layers (quaternion composition IS the nonlinear
transform), layer normalization (the sphere IS the constraint), and
most learned parameters (the algebra is exact).

## Open questions (revised)

**Emission threshold.** When does a level emit? If σ < ε, the
subsequence is "closed enough." ε determines granularity:
tight = character-level fragments, loose = long phrases. This
could be learned per level (one scalar per level) or fixed.

**Cross-level gradient flow.** σ at level 1 depends on closure
elements from level 0, which depend on the embedding table.
Gradient from L1's prediction loss should flow through L0's
emissions back to the embedding table. This requires the emission
operation to be differentiable — which it is, since it's just the
running product C at the threshold crossing.

**Anti-collapse through recursion.** Does the recursive structure
alone prevent degenerate embedding collapse? If level-1 prediction
loss forces character embeddings apart (because collapsed characters
produce indistinguishable words), then cross-entropy at level 0 may
be redundant. This determines the true minimum overhead.

**Factor specialization.** With m factors, do different factors
learn different aspects of language? Testable: after training,
freeze all but one factor and measure σ separation on different
linguistic tasks (syntax vs. semantics vs. phonetics).

**Level count.** How many levels does English need? Characters →
words → phrases → sentences → paragraphs. That's 4–5 levels. But
the algebra might discover a different hierarchy — maybe morphemes
are a level, maybe clauses are. The number of levels and the
emission thresholds together define the compositional granularity.

**Minimum overhead per level.** Level 0 needs a neural translator
(proven). Level 1+ receives closure elements that are already on
S³. Do they need a neural translator too, or is direct composition
sufficient? If level-1 tokens are already unit quaternions, the
"embedding" step is identity — no network needed. This would mean
the neural overhead exists only at level 0 (the boundary between
raw data and the geometry), and everything above is pure algebra.
