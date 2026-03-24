"""The chain — translates ball geometry into labeled color channels.

The ball holds full-color quaternions. This module is the prism that
splits them into human-readable channels using the Hopf fibration.

Four operations:

    expose(element)              — any point on the ball → Valence.
                                   Works at every step of composition,
                                   not just at incident time.

    incident_drift(inc, src, tgt) — the local gap at a specific incident.
                                   Takes the incident and both composed
                                   paths, returns the drift quaternion at
                                   that position. Feed this into
                                   expose_incident for per-incident color.

    expose_incident(inc, drift)  — a localized incident → IncidentValence.
                                   Same channels, plus structural labels:
                                   which positions, what payload, what broke.

    bind(a, b)                   — the connector. Two spheres → Binding.
                                   Checks both the equal and inverse
                                   relationships in one pass. Returns which
                                   holds (if either) and the gap's colors.

The 3+1 channels:

    W  (scalar, S¹ fiber)   — the coherence axis. Has or hasn't.
                               Parametrized by e (self-evident, axiom 2).
    R, G, B  (vector, S²)   — the interaction axes. Where and how far.
                               Parametrized by π (observed, axiom 1).

Two incident types, same algebraic object, different broken axis:

    Missing record      — W broke (existence). One position is None.
    Content mismatch    — RGB broke (position). Both positions present, different.

This module translates. It never decides which side is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hopf import hopf_decompose
from .canon import IncidentReport

# The S³ group object — needed for compose and inverse in bind().
import closure_rs
_SPHERE = closure_rs.sphere()


def _to_quaternion(element: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pad a closure element to a full quaternion for Hopf decomposition."""
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = min(len(element), 4)
    q[:n] = element[:n]
    return q


@dataclass(frozen=True)
class Valence:
    """The color channels of any point on the ball.

    sigma  — how far from identity (the magnitude dial).
    base   — direction on S² as (R, G, B). What kind of divergence.
    phase  — angle on S¹ as W. Which fiber.
    """

    sigma: float
    base: tuple[float, float, float]  # S² direction → (R, G, B)
    phase: float  # S¹ fiber → W


@dataclass(frozen=True)
class IncidentValence:
    """A fully labeled incident in color channels.

    Carries everything Valence has (sigma, base, phase) plus the
    structural context of the incident:

    position_a    — where in stream A (None if the record is absent from A).
    position_b    — where in stream B (None if the record is absent from B).
    payload       — the actual record bytes.
    axis          — "existence" (W broke, record missing) or
                    "position" (RGB broke, record moved).
    displacement  — how many positions apart (None if missing).
    """

    # From the algebra
    position_a: int | None
    position_b: int | None
    payload: bytes

    # Derived labels
    axis: str  # "existence" | "position"
    displacement: int | None

    # Hopf channels
    sigma: float
    base: tuple[float, float, float]  # S² direction → (R, G, B)
    phase: float  # S¹ fiber → W


def expose(element: NDArray[np.float64]) -> Valence:
    """The prism. Takes any point on the ball and splits it into color
    channels via Hopf. Call this after every ingest to watch the
    channels evolve in real time, or on any diff to see what kind of
    divergence it is.
    """
    q = _to_quaternion(element)
    hopf = hopf_decompose(q)

    return Valence(
        sigma=float(hopf["sigma"]),
        base=tuple(float(x) for x in hopf["base"]),
        phase=float(hopf["phase"]),
    )


def expose_incident(incident: IncidentReport, drift_element: NDArray[np.float64]) -> IncidentValence:
    """The labeler. Takes a localized incident and the drift quaternion,
    splits the quaternion into channels, and attaches structural labels:
    which axis broke, which positions, how far apart. This is the
    endpoint of the chain — what the application layer reads.
    """
    q = _to_quaternion(drift_element)
    hopf = hopf_decompose(q)

    if incident.source_index is not None and incident.target_index is not None:
        axis = "position"
        displacement = abs(incident.source_index - incident.target_index)
    else:
        axis = "existence"
        displacement = None

    return IncidentValence(
        position_a=incident.source_index,
        position_b=incident.target_index,
        payload=incident.record,
        axis=axis,
        displacement=displacement,
        sigma=float(hopf["sigma"]),
        base=tuple(float(x) for x in hopf["base"]),
        phase=float(hopf["phase"]),
    )


def incident_drift(
    incident: IncidentReport,
    source_path,
    target_path,
) -> NDArray[np.float64]:
    """Extract the local gap quaternion at an incident's position.

    For a reorder incident (both positions present), the drift is
    the divergence between the two paths at the incident's source
    position: inv(source_composition_at_i) · target_composition_at_i.

    For a missing incident (one position is None), the drift is the
    embedding of the missing record composed with the inverse of the
    path at the present position — what the gap looks like from the
    side that has the record.

    Returns a raw quaternion suitable for expose() or expose_incident().
    """
    si = incident.source_index
    ti = incident.target_index

    if si is not None and ti is not None:
        # Reorder: both sides present. Local gap at the source position.
        src_at = np.array(source_path.running_product(si + 1))
        tgt_at = np.array(target_path.running_product(ti + 1))
        gap = _SPHERE.compose(_SPHERE.inverse(src_at), tgt_at)
        return np.array(gap, dtype=np.float64)

    elif si is not None:
        # Missing from target. Use source composition at that point.
        src_at = np.array(source_path.running_product(si + 1))
        src_before = np.array(source_path.running_product(si))
        # The missing record's own contribution = inv(before) · at
        record_contrib = _SPHERE.compose(_SPHERE.inverse(src_before), src_at)
        return np.array(record_contrib, dtype=np.float64)

    else:
        # Missing from source. Use target composition at that point.
        tgt_at = np.array(target_path.running_product(ti + 1))
        tgt_before = np.array(target_path.running_product(ti))
        record_contrib = _SPHERE.compose(_SPHERE.inverse(tgt_before), tgt_at)
        return np.array(record_contrib, dtype=np.float64)


@dataclass(frozen=True)
class Binding:
    """The connector between two spheres.

    Takes two points on the ball and checks both relationships:
    equal (A · inv(B) = identity) and inverse (A · B = identity).

    relation  — "equal", "inverse", or "disordered".
    gap       — the Valence of whichever product was closer to identity.
    sigma     — the σ of that gap (0 = perfect match).
    """

    relation: str  # "equal" | "inverse" | "disordered"
    gap: Valence
    sigma: float


def bind(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    threshold: float = 1e-10,
) -> Binding:
    """The connector. Takes two points on the ball, computes both
    products — A · inv(B) and A · B — and determines the relationship
    between them.

    If the first product collapses to identity, the spheres are equal:
    they represent the same composition. If the second collapses to
    identity, they are inverses: one is the algebraic complement of
    the other. Otherwise the relationship is disordered, and the gap's
    valence describes its shape.
    """
    qa = _to_quaternion(a)
    qb = _to_quaternion(b)

    # Equal check: A · inv(B) → identity?
    inv_b = _SPHERE.inverse(qb)
    gap_eq = _SPHERE.compose(qa, inv_b)
    hopf_eq = hopf_decompose(np.array(gap_eq, dtype=np.float64))
    sigma_eq = float(hopf_eq["sigma"])

    # Inverse check: A · B → identity?
    gap_inv = _SPHERE.compose(qa, qb)
    hopf_inv = hopf_decompose(np.array(gap_inv, dtype=np.float64))
    sigma_inv = float(hopf_inv["sigma"])

    if sigma_eq < threshold:
        return Binding(
            relation="equal",
            gap=Valence(
                sigma=sigma_eq,
                base=tuple(float(x) for x in hopf_eq["base"]),
                phase=float(hopf_eq["phase"]),
            ),
            sigma=sigma_eq,
        )

    if sigma_inv < threshold:
        return Binding(
            relation="inverse",
            gap=Valence(
                sigma=sigma_inv,
                base=tuple(float(x) for x in hopf_inv["base"]),
                phase=float(hopf_inv["phase"]),
            ),
            sigma=sigma_inv,
        )

    # Disordered — return whichever gap was closer
    if sigma_eq <= sigma_inv:
        hopf_use, sigma_use = hopf_eq, sigma_eq
    else:
        hopf_use, sigma_use = hopf_inv, sigma_inv

    return Binding(
        relation="disordered",
        gap=Valence(
            sigma=sigma_use,
            base=tuple(float(x) for x in hopf_use["base"]),
            phase=float(hopf_use["phase"]),
        ),
        sigma=sigma_use,
    )
