# Gilgamesh Learns Walking

## The Principle

Gilgamesh is thrown into a body.  It starts as ONE cell.  That cell reads the body's orientation on S³ and tries to maintain identity (upright).  When it encounters a problem it cannot close alone, it spawns a child cell to handle that problem.  The body's joint structure is DISCOVERED through this process, not pre-wired.

This is the same mechanism that makes the kernel discover bar structure in music and phrase structure from bar emissions.  The adapter provides the binary truth (which joints exist, what their angles are, whether feet are touching ground).  The lattice discovers what to DO with them.

## How It Works

### Step 1: One Cell, One Body

Gilgamesh starts as a single Enkidu cell.  Its state is the body's torso orientation quaternion — already on S³.  Identity = upright.  σ = tilt from vertical.

The cell's only action: set ALL joints to the position that its C⁻¹ suggests.  C⁻¹ is a quaternion.  The pitch component (i-axis) drives all hip joints.  The roll component (j-axis) drives all ankle/abduction joints.  Every joint gets the same correction.

This is crude.  All joints move together.  But it's enough to stand — the body-level C⁻¹ keeps the torso roughly upright by pushing all joints in the correction direction.  This is what Phase 1 (stand) already demonstrated.

### Step 2: Problems Create Children

When the single cell tries to walk (lean forward from drive pressure), σ oscillates.  The oscillation has STRUCTURE: sometimes the pitch component spikes (falling forward/back), sometimes the roll component spikes (falling sideways).  These are different problems on different axes.

The binary rule: the cell decomposes the problem via Hopf.  W-dominant gap = missing (something isn't there that should be).  RGB-dominant gap = reorder (something is in the wrong arrangement).

When a SPECIFIC axis consistently fails to close, the cell spawns a child cell to handle that axis.  The child cell receives events from that axis only and specializes in closing them.

Concretely:
- If pitch (i-axis) consistently fails → spawn a PITCH cell that controls the sagittal hip joints
- If roll (j-axis) consistently fails → spawn a ROLL cell that controls the lateral hip/ankle joints
- If a pitch cell's σ oscillates between left and right legs → spawn LEFT and RIGHT children

### Step 3: The Hierarchy Emerges

```
Gilgamesh (body cell)
│  State: torso orientation
│  Drives: hunger, cold
│  Spawns children when it can't close an axis alone
│
├── Pitch cell (spawned when forward/back balance fails)
│   │  Controls: sagittal hip joints
│   │  Events: pitch component of body orientation
│   │
│   ├── Left leg cell (spawned when pitch cell can't close both legs at once)
│   │   Controls: left hip, left knee
│   │
│   └── Right leg cell (spawned when anti-phase is needed)
│       Controls: right hip, right knee
│
└── Roll cell (spawned when lateral balance fails)
    Controls: hip abduction / ankle joints
```

This hierarchy is NOT hardcoded.  It EMERGES from the body's failure modes.  A body with different joints would produce a different hierarchy.  A quadruped would spawn four leg cells.  A snake would spawn many segment cells.  Gilgamesh doesn't know the body — it discovers it.

### Step 4: Walking Emerges

Walking is what happens when the pitch cell's children discover anti-phase oscillation.

The body cell leans forward (drive creates σ on the forward axis).  The pitch cell tries to close it by adjusting the hips.  But one correction (both legs back) isn't enough — the body tips forward again immediately.  The pitch cell's σ oscillates.

When the pitch cell spawns left and right children, each child independently tries to close its own leg.  The left child pushes the left leg forward to catch the fall.  The right child pushes the right leg forward next.  They alternate because each one closes when its leg reaches its target, emits upward, and the OTHER child takes over.  The alternation IS the gait.

The knees flex during swing because the knee angle contributes to the pitch cell's σ — a straight swinging leg hits the ground, σ spikes, so the knee cell learns to flex during swing.  This is directional teaching: flex helped → reinforce.

### Step 5: Drives Navigate

Once walking works, the body cell's drives steer.  Hunger points toward food, cold points toward home.  The body cell's desired state tilts toward the drive target.  The pitch cell follows the tilt.  The leg cells alternate to execute it.  Navigation is just walking with a varying lean direction.

## Implementation Plan

### Phase 1: Body cell stands (DONE)
One cell.  C⁻¹ drives all joints uniformly.  The body stays upright.

### Phase 2: Body cell spawns pitch and roll children
The body cell monitors its σ decomposition.  When the pitch component consistently exceeds a threshold, it spawns a pitch cell.  When roll does, it spawns a roll cell.  Each child receives only its axis of the body state.

### Phase 3: Pitch cell spawns leg children
The pitch cell discovers it needs anti-phase.  It spawns left and right children.  Each leg child controls its own hip and knee.  The alternation emerges from closure cadence.

### Phase 4: Gait stabilizes
The directional teaching on each cell converges.  The genome stores: "for this body tilt, this joint angle helps."  The gait pattern persists in the genome.  The body walks indefinitely.

### Phase 5: Drives navigate
Food and shelter targets create lean.  The body walks toward targets.  Eats food.  Returns home.  Enkidu Alive in 3D.

## The Hopf Fiber for 3D Movement

The body's orientation quaternion decomposes exactly as the theory predicts:
- **W**: existence/coherence = upright or not.  σ from identity.
- **i-axis**: sagittal = forward/back tilt.  Drives hip pitch.
- **j-axis**: lateral = left/right tilt.  Drives hip abduction / ankles.
- **k-axis**: yaw = rotation.  Drives steering.

Each axis is a separate closure problem.  Each can have its own cell.  The Hopf fiber IS the joint coordination map.  The three axes map to three types of joints, and the lattice discovers this mapping by failing and spawning.

## What This Proves

If this works, the same architecture that learns Bach phrases from bar emissions also learns gait patterns from joint states.  The kernel is the same.  The Hopf decomposition is the same.  The closure dynamics are the same.  The only difference is the adapter (music notes vs joint angles) and the truth (barlines vs foot contacts).

The body doesn't know it has legs.  It discovers that it has legs because the pitch axis won't close without them.  This is the architecture learning the substrate, not the programmer encoding it.
