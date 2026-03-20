# Closure SDK

A suite of tools for composing and interacting with a primitive
data structure — the geometry of ordered information.

---

## What this is

A primitive data structure. Not a library built on top of hashes
or Merkle trees — a new fundamental structure, like a stack or a
queue or a blockchain, that everything else builds on.

A stack is LIFO, a queue is FIFO, a hash map is key-value, a
blockchain is a hash chain — all invented structures. This one
isn't invented. It is what happens when you compose ordered data
on the geometry that the math actually demands (S³, unit
quaternions, the richest space where sequential composition is
still associative). The structure falls out of two axioms and
Hurwitz's theorem. We implemented it.

Any data that can be serialized to bytes can be composed. Databases,
blockchains, gene sequences, financial ledgers, network packets,
satellite telemetry, event streams — anything ordered.

## What you get

    Instant coherence     — did my data change? One comparison, any scale.
    Exact magnitude       — how much changed, as a precise geometric distance.
    Incident type         — missing record (existence broke) or reorder
                            (position broke). Two types only, algebraic
                            inverses of each other. There is no third type.
    Localization          — which record, found in O(log n). At 1 million
                            records, 20 comparisons.
    Channel decomposition — the Hopf fibration splits every divergence into
                            3+1 channels: W (has or hasn't) + RGB (where
                            and how far). Available at every step.
    Composability         — the output of any composition is a valid input
                            to the next. Combine summaries across shards,
                            diff two snapshots, feed one check into another.
    Exact sensitivity     — perturb one record by ε, anywhere in the
                            sequence, the composition shifts by exactly ε.
                            First record or billionth. Proven (Theorem 1).
    Uniform visibility    — every position in the sequence is equally
                            detectable. Proven (Theorem 2).
    Real-time streaming   — classify incidents as records arrive, before
                            either stream is complete.
    Identity binding      — two systems prove they hold the same data by
                            comparing elements, never the data itself. One
                            call checks both products: equal, inverse, or
                            disordered.
    Cryptographic summary — embed() hashes with SHA-256 before composing.
                            The element is a 32-byte summary that cannot be
                            reversed to the original data, but can be
                            composed, diffed, inverted, and decomposed into
                            channels. A hash you can do algebra on.

## What problems this solves

    Byzantine fault detection          is A still A?
    Multi-fault localization           where did it break, for each break?
    Stream reconciliation              are these two live streams the same?
    The oracle problem                 is this record missing or just late?
    Tamper detection                   was this record altered?
    Ordered data verification          did this arrive in the right order?
    Replica consistency                do these two copies still match?
    Pipeline integrity                 did records get dropped, duplicated, or reordered?
    Cross-system reconciliation        do these two systems agree on what happened?
    Zero-knowledge verification        do we hold the same data, without revealing it?

## Original contributions

**Resolving ambiguity in space** (`Gilgamesh`).


Given two complete sequences, finds every point where they
diverged. Both sequences are already written — the question is
spatial: where, exactly, do they differ, and how many times? Compose
both on S³, and the geometry already encodes every divergence. Walk
the disagreement region once, ask one question per record — does the
other side have it, or doesn't it? — and the Hopf fiber classifies
the answer. O(n + log n).

A quantitative contribution — efficient solutions for comparing data
in files exist; the advance is precision and efficiency on a new
substrate. 

**Resolving ambiguity in time** (`Enkidu`).
When a record arrives on one side
with no match on the other, the question is temporal: is it absent,
or just late? The future is not yet written — no amount of
computation resolves this. The principle is simple: we can embed
exactly one semantic meaning into data — it either exists or it
doesn't. The rolling counter makes the absence explicit: hold the
record one cycle, and the empty slot where its match should be IS
the answer. The grace period turns the oracle problem into a bounded
binary decision. After any misalignment the next step is the same
case, the same question, the same tool. Errors chain, and the tool
chains with them. 

A qualitative contribution — no prior solution treats the oracle
problem as a geometric variable.

Both modes rest on the same geometric fact: the Hopf fiber guarantees
that missing and reorder are algebraic inverses. They are not labels
assigned by a classifier — they are the two axes the quaternion
provides, orthogonal and exhaustive.

**The Zeroth Law of Thermodynamics — The Law of Coherence (axis completeness).**

The universe provides exactly two ways for coherence to break:
something exists or it doesn't, and something is in the right place
or it isn't. Existence is a scalar question — present or absent,
the W axis, symmetric under inversion. Position is a vector question
— here or displaced, the RGB axes, antisymmetric under inversion.
These are the two axes quaternion algebra provides, and every
divergence between two ordered streams lands on one of them.

A quaternion q = w + xi + yj + zk has two parts that behave
differently under conjugation: the scalar w is symmetric (unchanged),
the vector (xi + yj + zk) is antisymmetric (sign flips). A divergence
between two compositions lands on one or the other — never both, never
neither. invert() flips one axis while preserving the other, which is
why the two incident types are algebraic inverses.

    Missing record — existence axis (W):
        stream A: [tx_001, tx_002, tx_003]
        stream B: [tx_001, tx_003]
        diff → W anomaly, RGB near zero.
        axis = "existence". position_b = None.

    Reorder — position axis (RGB):
        stream A: [tx_001, tx_002, tx_003]
        stream B: [tx_001, tx_003, tx_002]
        diff → RGB anomaly, W near zero.
        axis = "position". displacement = 1.

Theorems 1 and 2 measure how precisely and uniformly the SDK detects
divergence. Theorem 0 is what they are measuring: the two axes are
orthogonal, exhaustive, and geometrically stable — which is what
makes exact sensitivity and uniform detectability possible at all.

**The Hopf-to-color mapping** (`expose`, `expose_incident`).
The Hopf fibration decomposes S³ into S² × S¹. We map this to a
perceptual channel system analogous to the human visual system:
W (scalar, S¹ fiber) = luminance = existence (has or hasn't),
RGB (vector, S² base) = chrominance = displacement (where and how
far). A missing record shows up as a W anomaly — the light went out.
A reordered record shows up as an RGB anomaly — the light is there,
pointing the wrong way. This mapping is available at every step of
the composition, not only at incident time.

**The two-axiom derivation**.
Two axioms — existence requires interaction, coherence requires
directionality — produce the entire structure in a 1:1 mapping:

    Axioms → Lie bracket and exponential map
           → vector (i,j,k) and scalar (w) quaternion parts
           → Theorem 1 (exact sensitivity) and Theorem 2 (uniform detectability)
           → S³ properties (simply connected, non-commutative, compact)
           → SDK operations (embed, compose, invert, sigma, diff, compare)
           → Hopf channels (RGB from axiom 1, W from axiom 2)

The parametrization is Euler's identity: e^(iπ) + 1 = 0 —
coherence (e) flowing through interaction (π) via coupling (i)
produces inversion, and composing with identity gives closure.

---

## Architecture

```
THE SPHERE (S³ — where everything lives)
│
│  Every record becomes a point on this sphere.
│  Every composition is a movement on this sphere.
│  Identity (north pole) = perfect coherence.
│  Distance from identity = how much something broke.
│
├── ENTRY
│   └── embed()                          ops.py:30       bytes → point on the hypersphere
│
├── TOOLS (operate on points)
│   ├── compose(a, b)                    ops.py:41       multiply two points
│   ├── invert(a)                        ops.py:51       the opposite point (undo)
│   ├── sigma(a)                         ops.py:61       distance from north pole
│   ├── diff(a, b)                       ops.py:69       the gap between two points
│   └── compare(a, b)                    ops.py:79       gap + threshold → coherent?
│
├── LENSES (three focal lengths)         lenses.py
│   ├── Seer                             lenses.py:34    sensor, O(1), detects drift
│   ├── Oracle                           lenses.py:99    recorder, O(n), locates in O(log n)
│   └── Witness                          lenses.py:200   reference template, verifies
│
├── ANSWER FORMATS                       state.py
│   ├── ClosureState                     state.py:24     a point on the hypersphere
│   ├── CompareResult                    state.py:43     drift + coherent flag
│   └── LocalizationResult              state.py:58     position + steps
│
│
THE CANON (finds what broke — two modes, same classification)
│
│  When the sphere says "something diverged," the canon finds WHERE
│  and classifies WHAT: missing (W broke) or reorder (RGB broke).
│
├── STATIC MODE (both streams complete)
│   └── gilgamesh(source, target)     canon.py:280    compose, walk both chains, classify
│                                                        O(n + log n)
│
├── STREAM MODE (records arrive one at a time)
│   └── Enkidu                 canon.py:132    match, wait, promote, reclassify
│       ├── .ingest(payload, pos, side)                  classify on arrival
│       └── .advance_cycle()                             roll the counter, promote unresolved
│                                                        one binary question per tick:
│                                                        missing (payload) or reorder (position)?
│
├── IncidentReport                       canon.py:114    type + positions + payload + steps
│                                                        (missing = W broke, reorder = RGB broke)
│
└── RetentionWindow                      canon.py:72     evidence archive (raw records for investigation)
│
│
THE CHAIN (Hopf projection → color channels)
│
│  The sphere holds full-color quaternions. The chain is the prism
│  that splits them into human-readable channels.
│  W  (scalar)  = coherence axis    (e, self-evident, axiom 2)
│  RGB (vector) = interaction axes  (π, observed, axiom 1)
│
├── THE PRISM (internal)                 hopf.py
│   ├── hopf_project                     hopf.py:26      → S² direction (RGB)
│   ├── hopf_fiber_phase                 hopf.py:38      → S¹ angle (W)
│   └── hopf_decompose                   hopf.py:48      → σ + RGB + W
│
├── EXPOSE (public interface)            valence.py
│   ├── expose(element)                  valence.py:94   any point → Valence
│   ├── expose_incident(inc, drift)      valence.py:110  incident → IncidentValence
│   └── bind(a, b)                       valence.py:139  two points → Binding
│                                                        equal, inverse, or disordered
│
└── CHANNELS                             valence.py
    ├── Valence                          valence.py:51   σ + base(R,G,B) + phase(W)
    ├── IncidentValence                  valence.py:65   channels + positions + payload + axis
    └── Binding                          valence.py:139  relation + gap valence + σ


PIPELINE (data flows top to bottom)

    raw bytes
        │
        ▼
    embed()                  ops.py:30 ─────────── ENTRY
        │
        ▼
    Seer / Oracle / Witness  lenses.py ─────────── LENSES compose on the sphere
        │
        ▼
    sigma() / compare()      ops.py:61 / :79 ───── TOOLS measure on the sphere
        │
        ├── coherent? → done
        ├── bind(a, b) ─────── valence.py:139 ──── BINDING equal / inverse / disordered
        │
        ▼
    gilgamesh()  ─┐
                     ├─ canon.py ───────────────── CANON finds each incident
    Enkidu ┘    static (complete) or stream (online)
        │
        ▼
    expose_incident()        valence.py:110 ─────── CHAIN translates to channels
        │
        ▼
    IncidentValence          valence.py:65 ──────── labeled W + RGB + positions + axis
        │
        ▼
    (application layer)      ───────────────────── human or system decides


FRONT DOOR

    __init__.py              21 symbols exported
    closure_sdk/
      ops.py                 the sphere's tools (6)
      lenses.py              Seer, Oracle, Witness
      state.py               answer formats (3)
      canon.py               gilgamesh, Enkidu, IncidentReport, RetentionWindow
      valence.py             expose, expose_incident, bind, Valence, IncidentValence, Binding
      hopf.py                the prism (internal, wrapped by valence)
```

---

## Theory

### Why S³

Hurwitz's theorem: the normed division algebras are ℝ, ℂ, ℍ, 𝕆.
Each doubling trades a property:

    ℝ → ℂ    trades ordering
    ℂ → ℍ    trades commutativity
    ℍ → 𝕆    trades associativity

S³ (unit quaternions, ℍ) is the richest space where composition is
still associative — required for sequential running products,
(a·b)·c = a·(b·c). Trading commutativity means order matters:
the same record at position 5 and position 10 contributes differently.
This is what lets the SDK detect reordering.

Like choosing which number system to do accounting in. Real numbers
work but lose sign. Complex numbers keep sign but lose ordering.
Quaternions keep ordering but lose commutativity — and for
sequential data, commutativity is exactly what you want to lose,
because order matters. Octonions lose associativity, which breaks
running totals entirely. Quaternions are the sweet spot.

### Compose on S³, project down

S³ is the only composition space. Lower-dimensional geometries
(S1, T^k) see less of the same structure — S1 loses ordering,
T^k loses geometric coherence between channels. The architecture:

    1. Always embed on S³
    2. Always compose on S³
    3. Project down via the Hopf fibration when you need channels

The Hopf fibration (S³ → S² × S¹) decomposes any quaternion into:

    sigma   — geodesic distance from identity (how much diverged)
    base    — S² point as (R, G, B) direction (what kind of divergence)
    phase   — S¹ angle as W (which coherence fiber)

This projection is available at every step of the composition.
It replaces per-channel accounting with geometrically coherent
channel decomposition — more information, and the channels talk
to each other through the quaternion structure.

Note: S1/T^k remain available in the Rust engine (`closure_rs`) for
use cases that genuinely need them (pure ledger balance, explicit
k-channel accounting). The engine supports them; the SDK surfaces
only S³ because it is the complete view.

### The 3+1 channels

The Hopf decomposition maps to a 3+1 hierarchy:

    W  (scalar, S¹ fiber)    existence / luminance     has ↔ hasn't
    R  (vector, S² base[0])  ┐
    G  (vector, S² base[1])  ├ displacement / chrominance
    B  (vector, S² base[2])  ┘

W is inverse to (R, G, B) — scalar vs vector in quaternion algebra.

Like splitting white light through a prism. The sphere holds the full
color. The chain (Hopf projection) separates it into a brightness
channel (W — is the light there at all?) and three color channels
(RGB — what direction is the light pointing?). A missing record
shows up as a W anomaly (the light went out). A reordered record
shows up as an RGB anomaly (the light is there but pointing the
wrong way).

### Why 3+1 — from two axioms

The 3+1 structure falls out of two axioms:

    Axiom 1: Existence requires interaction
    Axiom 2: Coherence requires directionality

#### Axioms → Lie group structure

    Axiom 1 (interaction)     →  The Lie bracket [X,Y] = XY - YX
                                  Generators defined entirely by how
                                  they interact. Pure relational
                                  structure.

    Axiom 2 (directionality)  →  The exponential map exp: g → G
                                  One-way flow from algebra to group.
                                  The arrow that makes composition
                                  coherent.

    Both together             →  The Jacobi identity
                                  [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
                                  Interactions cycle without accumulating
                                  contradiction. The Russell stopper.

#### Axioms → Quaternion parts

    Axiom 1 → vector part (i, j, k)
        Three imaginary axes, mutually generating: i·j = k, j·k = i,
        k·i = j. Each exists only through interaction with the others.
        Fully symmetric. Axiom 1 made algebraic.

    Axiom 2 → scalar part (w)
        ijk = -1 — the product of all three interactions lands on the
        real axis and stays there. The real axis receives but does not
        generate. Unidirectional flow. Axiom 2 made algebraic.

    The scalar part is what makes ℍ a division algebra — every nonzero
    element has a multiplicative inverse. Invertibility IS the coherence
    condition. The SDK's invert() depends on this.

#### Axioms → The two theorems

    Axiom 1 → Theorem 2 (uniform detectability)
        Jacobian = 1 everywhere. Every position equally detectable.
        S³ is simply connected (every point interacts with every other)
        and the metric is bi-invariant (interaction is symmetric), so
        every perturbation is visible from every vantage point.

    Axiom 2 → Theorem 1 (exact perturbation sensitivity)
        Perturb any event by ε — first or billionth — the summary
        shifts by exactly ε. The bi-invariant metric guarantees
        coherent measurement. Non-commutativity ensures WHERE the
        perturbation occurs is encoded in the result.

#### Axioms → Properties of S³

    Simply connected    ← Axiom 1: every point interacts with every
                          other. No topological hiding places.

    Non-commutative     ← Axiom 2: order matters. Same record at
                          position 5 and position 10 contributes
                          differently. Direction is baked in.

    Compact             ← Both: the space is finite and closed.
                          Closure is possible.

    Bi-invariant metric ← Both: measurement is frame-independent.

#### Axioms → SDK operations

    Axiom 1 (interaction):
        embed()    — any bytes enter the space
        compose()  — the interaction itself
        diff()     — the algebraic gap between two interactions

    Axiom 2 (coherence):
        invert()   — the undo that enables closure
        sigma()    — how far from identity
        compare()  — the verdict

    expose()       — the Hopf projection separates axiom 1 (RGB,
                     vector, interaction) from axiom 2 (W, scalar,
                     coherence)

#### The parametrization

The two axioms have natural constants:

    e  — self-evident. The function that equals its own derivative.
         Self-generating. Parametrizes the scalar axis (coherence,
         flow, W channel).

    π  — observed. Exists only as a ratio between two geometric
         quantities. Irreducibly relational. Parametrizes the vector
         axes (interaction, rotation, RGB channels).

    i  — the coupling between the two.

Euler's identity, e^(iπ) + 1 = 0: coherence (e) flowing through
interaction (π) via coupling (i) produces inversion, and composing
with identity gives closure. The SDK's pipeline — embed, compose,
invert, check identity — is this equation applied to ordered data.

### The structure of incidents

When the algebra localizes a divergence, it produces an IncidentReport
with three data points:

    position_a    where in stream A (None if absent)
    position_b    where in stream B (None if absent)
    payload       the displaced record bytes

Two incident types, derived from the data:

**Missing record** — existence axis broke (W-dominant).
One stream has the record, the other doesn't. Signature: one position
is None. Like a receipt that exists in one ledger but is absent from
the other — the question is about existence.

**Reorder** — position axis broke (RGB-dominant).
Both streams have the record, but at different positions. Signature:
both positions present, they differ. Like two copies of the same
receipt filed in different places — the question is about ordering.

Both are the same algebraic object: one axis symmetric, one
antisymmetric. They are semantic inverses of each other.

### The oracle problem

The algebra sees the structure of disagreement but cannot judge which
side is correct. The SDK decomposes and labels. The application layer
decides. Fault assignment is a policy decision that belongs upstream.

This is how real systems actually work. The immune system checks
whether the composition closes — it doesn't vote on which cell is
the "real" one. DNA repair checks whether the double helix closes —
it doesn't reason about which strand is authoritative.

Closure gives engineered systems the same ability. The algebra detects
and localizes divergence with exact sensitivity (Theorems 1 and 2).
The application layer decides what to do about it.

### The cryptographic property

The SDK's security rests on a two-layer pipeline:

    Layer 1: embed() hashes raw bytes with SHA-256 before placing
    them on the sphere. After this step, the original content is
    irreversible — no one can recover what the bytes were.

    Layer 2: all subsequent operations — compose, invert, diff,
    expose, bind — act on these hashed elements using the algebraic
    structure of S³. The operations manipulate the geometry without
    ever touching the original data, which was discarded at the
    hash boundary.

This gives the SDK a property known as homomorphic hashing: the
hash is cryptographically hard (SHA-256 preimage resistance) AND
algebraically manipulable (S³ group operations). Most hashes are
one or the other. SHA-256 is hard but opaque — you can compare
two hashes and nothing else. Algebraic structures are rich but
forgeable. The SDK composes one into the other: hash first, then
compose on the sphere.

The consequence for trust: two systems that independently compose
the same data will land on the same point on S³. They can share
their elements and call bind() to verify the relationship. If one
system altered its data, the element changes (SHA-256 guarantees
this), and bind returns disordered instead of equal. The geometry
is deterministic from the hash — you cannot produce an element
that binds equal to another's without holding the same data.

The consequence for privacy: only the element crosses the wire,
never the data. The element is 32 bytes regardless of how many
records were composed. A million records and a hundred records
produce the same-sized summary. The recipient learns whether the
data matches, not what the data is.

**Non-hashed mode.** The default pipeline always applies SHA-256.
For internal use cases where cryptographic security is not needed
— testing, simulation, or composing pre-hashed elements — the
Rust engine accepts raw elements directly via closure_rs. The
SDK's Python surface (embed, Seer, Oracle, Witness) always hashes.
The choice is at the entry point, not the algebra.

### The chaining insight (future)

Each S³ composition produces a Valence output (W, RGB). Multiple
compositions can be chained by mapping valence output to input
constraints. The colors become the linking interface. This is
quaternion composition in the decomposed basis.

---

## API Reference

### The sphere's tools — ops.py

Six operations on S³. Three from axiom 1 (interaction), three from
axiom 2 (coherence).

    embed(record: bytes) → ClosureState

        The entry door. Takes any raw bytes — a row, a transaction, a
        message — hashes them (SHA-256), and places the result on S³ as
        a unit quaternion. Deterministic: same bytes always land on the
        same point. After this, the SDK only knows geometry.

        Like stamping a fingerprint onto a globe. Same document, same
        spot, every time.

    compose(left: ClosureState, right: ClosureState) → ClosureState

        Quaternion multiplication: left · right. The running product.
        This IS the closure operation. Non-commutative: order matters,
        so compose(a, b) lands on a different point than compose(b, a).

        Like adding the next step to a running total, except the total
        lives on a sphere and remembers what order the steps came in.

    invert(state: ClosureState) → ClosureState

        Group inverse: a⁻¹, such that compose(a, invert(a)) = identity.
        The undo operation. Generates the duality between the two
        incident types — inverting one axis while preserving the other.

        Like finding the exact opposite of a step so that taking both
        returns you to the starting point.

    sigma(state: ClosureState) → float

        Geodesic distance from identity on S³. The thermometer.
        σ = 0 means the composition is clean (you're at the north pole).
        σ > 0 means something diverged by exactly that much.

        Like checking how far your running total is from zero. Zero
        means everything balanced out perfectly.

    diff(a: ClosureState, b: ClosureState) → ClosureState

        Algebraic difference: a⁻¹ · b. The gap between two points.
        Returns the state that, when composed with a, gives b.
        Equivalent to compose(invert(a), b).

        Like subtracting one running total from another — the result
        is the discrepancy itself, with direction.

    compare(a: ClosureState, b: ClosureState) → CompareResult

        Computes diff, measures sigma, returns drift + coherent flag.
        The quick verdict: are these two compositions the same?

        Like asking "do these two ledgers agree?" and getting back
        both the yes/no answer and how far off they are.

### Lenses — lenses.py

Three instruments for observing the composition. Same sphere, different
resolution.

    Seer()

        The sensor. Holds one running product. Constant memory, O(1).
        Feed it records, ask if the streams still match. Detects drift
        but cannot say where — like a smoke detector that tells you
        there's fire but not which room.

        .ingest(record: bytes)          feed one record
        .ingest_many(records)           feed many at once
        .drift() → float               sigma of current point
        .state() → ClosureState         snapshot the current point
        .compare(other) → CompareResult compare two sensors
        .reset()                        back to identity

    Oracle()

        The recorder. Stores every intermediate point. O(n) memory,
        O(log n) localization via binary search. You consult the Oracle
        when the Seer raises an alarm and you need to find where the
        divergence started — like a flight recorder that logs every
        moment so investigators can find the exact second things went
        wrong.

        .from_records(records) → Oracle         build all at once
        .append(record)                         build incrementally
        .check_global() → float                 sigma of final product
        .check_range(i, j) → float              sigma of sub-segment
        .recover(t) → ndarray                   element at position t
        .localize_against(other) → LocalizationResult
        .state() → ClosureState
        .compare(other) → CompareResult

    Witness()

        The reference template. Built once from known-good data. Checks
        any test data against the reference and finds where corruption
        starts — like a master key that you press new copies against to
        see where the grooves don't match.

        .from_records(records) → Witness        build from reference
        .check(test_data) → float               drift against reference
        .localize(test_data) → LocalizationResult
        .state() → ClosureState

### The canon — canon.py

Two composition modes, same classification. Finds where streams
diverge and classifies each incident as missing or reorder.

    gilgamesh(source: list[bytes], target: list[bytes]) → list[IncidentReport]

        Static mode. Both streams are complete. Gilgamesh has the full
        picture — he composed both sequences on S³ before walking, so
        he knows every record's position on both sides before asking
        the question. The inverse of Enkidu: where Enkidu discovers
        the answer after a grace period, gilgamesh knows it immediately.

        1. Compose & localize — embed both sequences on S³, build both
           paths as prefix products, binary-search for the first
           divergence. O(n) embed, O(log n) search. If σ ≈ 0, the
           streams agree — return empty.

        2. Walk both chains — two pointers, one on each side, advance
           in lockstep. At each step, the Hopf fiber classifies the
           record. Three outcomes, no others:

           Coheres — same payload at both pointers. Advance both.

           Missing — payload at one pointer does not exist on the other
           side. The W axis broke (existence). Flag it. Advance only
           the pointer that has the record — the other side has no gap,
           it just continues. This is the Enkidu move: emptiness in
           static data collapses the chain, so one pointer falls behind.

           Reorder — both payloads exist on the other side, but at
           different positions. The RGB axis broke (position). The
           geometry tells us which record is displaced: look up where
           each payload actually lives on the other side and compare
           distances. The one further from where it should be is the
           displaced record. Flag it with its actual positions. If both
           are equidistant, it is a swap — flag both, advance both.

        Same binary question as Enkidu — existence or position — but
        answered immediately from the composed picture. Same two types.
        Inverse resolution strategy. O(n + log n).

    Enkidu()

        Stream mode. The online matcher. Records arrive one at a time.
        Each unmatched record starts as potentially missing. The grace
        period turns the uncertainty into a bounded binary decision:
        missing (payload invertibility) or reorder (position invertibility).

        .ingest(payload, position, side) → IncidentReport | None
            Classify one record on arrival. Returns an IncidentReport
            only on reclassification (missing → reorder). New missing
            incidents are created by advance_cycle, because at arrival
            time we don't yet know if the record is truly missing or
            just late.

        .advance_cycle() → list[IncidentReport]
            Roll the counter. Every record that survived a full cycle
            without a match is promoted to missing. Returns the list
            of new missing incidents.

        .unresolved_source → int        pending source records
        .unresolved_target → int        pending target records
        .reclassified_count → int       missing→reorder count
        .reset()                        clear all state

    IncidentReport (frozen dataclass)

        incident_type: str              "missing" or "reorder"
        source_index: int | None        position in stream A
        target_index: int | None        position in stream B
        record: bytes                   the payload
        checks: int                     binary search steps

    RetentionWindow(maxlen: int)

        The evidence archive. A bounded rolling buffer that stores raw
        records so the canon can investigate when the Seer detects drift.

        .append(block_number, records)
        .flatten() → list[bytes]
        .total_records → int

### The chain — valence.py

The prism that splits sphere geometry into human-readable color channels.

    expose(element: NDArray) → Valence

        Takes any point on the sphere and decomposes it into 3+1 channels
        via the Hopf fibration. Works at every step of the composition —
        call it after every ingest to watch the channels evolve in real
        time, or on any diff to see what kind of divergence it is.

    expose_incident(incident: IncidentReport, drift: NDArray) → IncidentValence

        Takes a localized incident and the drift quaternion, decomposes
        into channels, and attaches structural labels: which axis broke,
        which positions, how far apart. This is the endpoint of the
        chain — what the application layer reads.

    bind(a: NDArray, b: NDArray) → Binding

        The connector. Takes two points on the ball, computes both
        products — A · inv(B) and A · B — and determines the
        relationship between them.

        If the first product collapses to identity, the spheres are
        equal: they represent the same composition. If the second
        product collapses to identity, they are inverses: one is the
        algebraic complement of the other. Otherwise the relationship
        is disordered, and the gap's valence describes its shape.

        Two systems prove they hold the same data when their spheres
        bind equal — only elements cross the wire, never the data
        itself. A system proves it holds another's data by producing
        the element that binds inverse. In a pipeline, binding each
        consecutive pair verifies the whole chain.

    Valence (frozen dataclass)

        sigma: float                    magnitude (how much)
        base: (float, float, float)     S² direction (R, G, B)
        phase: float                    S¹ fiber (W)

    IncidentValence (frozen dataclass)

        position_a: int | None          where in stream A
        position_b: int | None          where in stream B
        payload: bytes                  the record
        axis: str                       "existence" or "position"
        displacement: int | None        |pos_a - pos_b|
        sigma: float                    divergence magnitude
        base: (float, float, float)     divergence direction (R, G, B)
        phase: float                    divergence channel (W)

    Binding (frozen dataclass)

        relation: str                   "equal", "inverse", or "disordered"
        gap: Valence                    the color channels of the gap
        sigma: float                    magnitude of the gap (0 = perfect)

### Answer formats — state.py

The shapes that come back from the tools.

    ClosureState

        A point on the hypersphere. What you get from embed(), compose(), or
        any lens's .state() method.

        group: str                      always "Sphere"
        element: NDArray[float64]       the quaternion (4 floats)
        dim: int                        always 4

    CompareResult

        The verdict from comparing two points.

        drift: float                    geodesic distance
        coherent: bool                  drift < threshold

    LocalizationResult

        The answer from binary search — where the divergence starts.

        index: int                      position of first divergence
        checks: int                     binary search steps taken

---

## Invariants

**Canonical equivalence is byte equality.**
Same bytes → same group element → same contribution. The SDK does
not normalize. If upstream emits byte-different but semantically
equivalent payloads, the adapter must canonicalize before ingest.

Like two receipts for the same purchase written in different
handwriting — they mean the same thing, but they look different.
The SDK sees bytes, so the adapter must ensure identical meaning
produces identical bytes.

**Exchange symmetry for identical payloads.**
Identical bytes are interchangeable — the algebra cannot recover
identity the data never carried. If copies must be distinguished,
the adapter includes a signifier (sequence number, UUID) in the
payload. This is a property of the geometry.

Concretely: in stream mode, the hash-set matching stores one entry
per payload. In static mode, gilgamesh walks both chains with two
pointers and uses payload counters to classify each mismatch — the
W axis (existence) tells it whether a record is missing, and the RGB
axis (position) tells it whether a record moved. Both modes are
correct if the adapter enforces payload uniqueness. If it doesn't,
identical records pair arbitrarily — which is mathematically correct
(they are the same element) but may not match the application's intent.

Like two identical $20 bills — they're interchangeable. If you
need to track which specific bill, stamp a serial number on it.

**Position is unique in the composition.**
The composition is non-abelian: same bytes at position 5 and
position 10 produce different contributions. Position is baked
into the algebra.

---

## Upstream contract

The valence layer is the communication endpoint. It exposes a fixed
set of fields with deterministic derivation rules.

**Per-step (any closure element, any time):**

    sigma   float        geodesic distance from identity
    base    (R, G, B)    S² direction
    phase   float        S¹ fiber angle

**Per-incident (localized divergence):**

    position_a    int | None     where in stream A
    position_b    int | None     where in stream B
    payload       bytes          the record
    axis          str            "existence" or "position"
    displacement  int | None     |pos_a - pos_b|
    sigma         float          divergence magnitude
    base          (R, G, B)      divergence direction
    phase         float          divergence channel

**Per-binding (two elements compared):**

    relation      str            "equal", "inverse", or "disordered"
    gap           Valence        the color channels of the gap between them
    sigma         float          magnitude of the gap (0 = perfect match)

**Derivation rules (deterministic, never configurable):**
- axis = "existence" if either position is None, else "position"
- displacement = |position_a - position_b| if both present, else None
- sigma/base/phase from Hopf decomposition of the drift quaternion
- relation = "equal" if σ(A·inv(B)) < threshold, "inverse" if σ(A·B) < threshold, else "disordered"

The application layer chooses interpretation. The SDK provides structure.
