# Brahman Visual — Motion Learning on S³

Side project. Same S³ Transformer architecture as the language path, applied to spatial navigation. Separate from Brahman's LLM development.

## The principle

The same principle drives all three Brahman domains — brackets, language, motion:

1. Enkidu eats tokens (brackets, characters, actions)
2. Each token composes as a quaternion on S³
3. The running product drifts from identity → σ increases
4. The model's drive is to get closure back → reduce σ

This is FEP. σ IS the free energy. Every action either reduces σ (expected, low surprise) or increases it (surprising). The model learns to act to minimize surprise — to restore its expected state. The gradient of σ with respect to the embedding IS the learning signal.

For brackets: closure = balanced parentheses. The model discovers ( and ) are inverses.
For language: closure = coherent text. The model discovers character compositions that reduce σ.
For motion: closure = reaching the target (or returning home). The model discovers that actions compose spatially.

The architecture doesn't change. The loss doesn't change. Only the tokens change.

## Why this exists

Brackets proved S³ learns algebraic inverses. Characters test whether it learns linguistic structure. Motion tests something more fundamental: can S³ learn to navigate a space where the geometry IS the physics?

3D rotations are quaternions. This isn't an encoding — it's identity. The model doesn't need to discover that rotations compose; the architecture composes them natively. The question is whether it learns to PLAN compositions that achieve specific goals.

## Two domains

### Domain A: Rotation navigation (native S³)

The agent navigates orientations in 3D. This is the mathematically clean version — rotations ARE quaternions, no encoding.

- **Vocabulary:** 6 discrete rotations — ±x, ±y, ±z by a fixed angle (π/6)
- **Each step:** a unit quaternion (the actual rotation it represents)
- **Path:** composition of rotations = running product on S³
- **σ:** geodesic distance from current orientation to target
- **Closure:** σ → 0 means the agent reached the target orientation

Non-commutativity matters here: +x then +y ≠ +y then +x. The model must learn this from data. Brackets commute in the sense that (()) and (()) are both valid — rotations genuinely don't commute. This is a harder test.

### Domain B: Grid navigation

A 10×10 grid, agent walks to a target.

- **Vocabulary:** 5 tokens — {UP=0, DOWN=1, LEFT=2, RIGHT=3, EOS=4}
- **Each step:** a learned quaternion embedding (the model discovers the spatial structure)
- **Path:** composition of action embeddings
- **σ:** distance from target state
- **Closure:** σ → 0 means the agent reached the target cell

Grid actions commute (up-right = right-up for displacement), so this is easier than Domain A. Both domains use the same S3Transformer. Both test the same principle.

## How it works, mechanically

### How actions become quaternions

Each action is a token. The embedding layer maps each token to a unit quaternion on S³ (learned, same as brackets). The model doesn't know that UP is the inverse of DOWN — it discovers this from data, the same way it discovered ( is the inverse of ).

After training, UP and DOWN should embed as quaternions whose product is identity. Same for LEFT/RIGHT. For rotations, +x and -x should be inverses. The model learns spatial inverses from data and geometry alone.

### What the hidden state means

At each position in the sequence, the model's hidden state is a unit quaternion. After attention and composition through the layers, this quaternion represents the model's geometric understanding of "where am I given this action history."

The hidden state ISN'T an opaque vector. It's a point on S³ with measurable properties:
- **σ** = geodesic distance from identity = "how far am I from where I started"
- **Hopf channels** (R, G, B, W) = directional decomposition = "WHICH direction am I offset, and how"

These are available at every step, for free. The model navigates with a built-in compass.

### The FEP loop

At every step of generation:
1. Model takes an action → token gets composed into the running product
2. σ changes — did the action bring us closer to closure or farther?
3. Hopf channels change — which direction are we drifting?
4. This information is available to the model at the next step
5. The model's prediction adjusts to reduce σ

With valence feedback (Step 2 of Brahman, already proven on brackets): the full Hopf decomposition (σ, R, G, B, W) feeds back into the embedding at each step. The model doesn't just know it's drifting — it knows which way, and it steers.

### How training works

Supervised, not RL. Same as brackets:

1. **Generate expert data:** BFS shortest paths (grid) or geodesic decompositions (rotations). Each trajectory is a sequence of action tokens ending with EOS.

2. **Training objective:** Next-action prediction (cross-entropy). Given action history [UP, UP, RIGHT, ...], predict the next action.

3. **Closure loss:** σ_final for paths that reach the target. The prediction loss teaches WHAT to do, the closure loss teaches that the path should CLOSE.

4. **What the model learns:**
   - Spatial inverses (UP/DOWN, LEFT/RIGHT, +x/-x) from composition
   - Path planning (which actions compose to reach a target)
   - Efficiency (shorter paths have lower cumulative σ drift)

### Inference

Autoregressive generation, identical to bracket generation:
1. Encode the target as a conditioning signal (prepended to the sequence or as initial state)
2. Generate one action at a time
3. At each step, σ tells the model how close it is to the target
4. Hopf channels tell it which direction to go
5. Stop when the model predicts EOS

### Where Enkidu fits

Enkidu watches the generated path against the expert (reference) path in real time:
- **Matched:** action matches the reference → agent is on track
- **Missing:** agent skipped an action the expert took → gap in the path
- **Reorder:** agent took the right actions in wrong order → took a different route
- **Extra:** agent invented an action not in the reference → off-path

The agent gets Enkidu's classification as feedback. Not just "you're drifting" (σ) but "you skipped step 4" or "steps 3 and 5 are swapped." This is the external monitoring loop applied to spatial planning — the first real test of Seer + Enkidu in a generative context.

## Architecture

```
S3Transformer(
    vocab_size  = 5,     # grid: UP, DOWN, LEFT, RIGHT, EOS
    # or
    vocab_size  = 7,     # rotations: +x, -x, +y, -y, +z, -z, EOS
    m_factors   = 2,     # (S³)² = 8D — one factor per spatial axis
    n_layers    = 4,
    hidden      = 64,    # MLP width in embed/head
    max_seq_len = 64,    # max path length
)

Estimated parameters: ~2,500 (grid) / ~3,000 (rotations)
```

### Why m=2, not m=1

**First run (m=1, 849 params): σ separation PASSED (t=42.65) but generation FAILED (9.3% closed).**

The model could recognize closure but couldn't produce it. The reason: grid walking has two independent degrees of freedom (x and y displacement). A single S³ factor (4D) has one compositional axis — it can track one direction but loses the other. Example from the run: `↓↓→↑←` had σ=0.0136 (near zero) despite NOT being closed — the model collapsed two dimensions into one and lost information.

Brackets worked with m=1 because bracket depth is one degree of freedom. Grid walking needs m≥2: one S³ factor to track x displacement, one for y. By the same logic, 3D rotations (Domain A) need m≥3.

This is actually a meaningful finding: **the minimum number of S³ factors equals the degrees of freedom in the compositional task.** Brackets = 1 DOF → m=1. Grid = 2 DOF → m=2. 3D rotations = 3 DOF → m=3. Language = ? DOF → the dimensionality experiment answers this.

Same class as `brahman/model.py:S3Transformer`. No new architecture code — just new data generation and training scripts.

## Data generation

```python
# Grid
def generate_expert_path(grid_size, start, target):
    """BFS shortest path. Returns action tokens + EOS."""

# Rotations
def generate_rotation_path(target_quaternion, max_steps):
    """Greedy geodesic decomposition into discrete rotations + EOS."""
```

Training set: 50,000 random trajectories.
Validation set: 2,000 trajectories.

## Evaluation

### 1. Path validity
Generate 1,000 paths. What fraction reach the target?

### 2. σ separation
σ_final of successful vs failed paths. If σ separates them, the geometry captures navigation success.

### 3. Inverse discovery
After training: does embed(UP) · embed(DOWN) ≈ identity? The model should discover spatial inverses from data alone.

### 4. Generalization
Train on small grids (up to 8×8). Test on larger (10×10, 15×15). Does composition scale?

## Visualization

1. **Grid trace:** Agent's path colored by σ at each step (red = far, green = close).
2. **Sphere trace:** Running product projected to 3D via stereographic projection. Path on the sphere.
3. **σ over time:** Line plot showing σ decreasing as agent approaches target.
4. **Learning progression:** Same start/target across epochs. Random walk → correct direction → optimal path.
5. **Embedding geometry:** Action embeddings plotted on S³. UP/DOWN should be antipodal. LEFT/RIGHT antipodal.
6. **Enkidu overlay:** Red markers on the path where Enkidu classifies incidents.

Tools: `matplotlib.animation` for GIFs, `plotly` for interactive 3D.

## Compute

CPU only. ~800 parameters, 5-7 token vocab, short sequences (max ~20 actions). Training: seconds to minutes.

## File structure

```
brahman/visual/
    __init__.py
    BRAHMAN_VISUAL.md      ← this file
    data.py                ← grid/rotation environment, BFS, data generation
    train.py               ← training loop, evaluation
    visualize.py           ← path animation, embedding plots, σ traces
    enkidu_loop.py         ← Enkidu monitoring integration
```

## Status

Spec complete. Not built yet.
