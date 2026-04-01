"""
S³ geometry utilities.

geodesic_step: SLERP from a toward b on S³ by fraction t.
hopf_classify: classify a gap quaternion as 'missing' (W-dominant) or 'reorder' (RGB-dominant).
"""

import math
import numpy as np


def geodesic_step(a: np.ndarray, b: np.ndarray, t: float = 0.1) -> np.ndarray:
    """SLERP from a toward b on S³ by fraction t."""
    d = float(np.dot(a, b))
    if d < 0:
        b = -b
        d = -d
    if d > 0.9999:
        return b.copy()
    theta = math.acos(min(d, 1.0))
    s = math.sin(theta)
    if s < 1e-8:
        return a.copy()
    r = math.sin((1 - t) * theta) / s * a + math.sin(t * theta) / s * b
    n = np.linalg.norm(r)
    return r / n if n > 1e-8 else a.copy()


def hopf_classify(gap_q: np.ndarray) -> str:
    """Classify a gap as 'missing' (W-dominant) or 'reorder' (RGB-dominant).

    W axis (scalar) encodes existence — something absent.
    RGB axes (vector) encode structure — something misplaced.
    """
    w = abs(float(gap_q[0]))
    rgb = math.sqrt(float(gap_q[1])**2 + float(gap_q[2])**2 + float(gap_q[3])**2)
    return 'missing' if w > rgb else 'reorder'