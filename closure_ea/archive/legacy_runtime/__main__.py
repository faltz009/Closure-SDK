"""Entry point: python -m closure_ea.

Closure EA is the learning-runtime layer built on the same S^3 algebra used by
the SDK. The core runtime lives in:

- kernel.py  — substrate-agnostic closure recurrence on S^3
- cell.py    — one adaptive closure cell: adapter + kernel + local genome
- lattice.py — multi-level living lattice of closure cells
- trinity.py — the S1/S2/S3 learning loop
- genome.py  — persistent learned structure saved to disk

Tracks provide substrate-specific adapters and corpora on top of that core.
"""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    print("Closure Ea")
    print(f"  Refactor spec: {root / 'BRAHMAN_REFACTOR_IMPLEMENTATION.md'}")
    print(f"  Core runtime:  {root / 'kernel.py'}")
    print(f"                 {root / 'cell.py'}")
    print(f"                 {root / 'lattice.py'}")
    print(f"                 {root / 'trinity.py'}")
    print(f"                 {root / 'genome.py'}")
    print(f"  Enkidu Alive:  {root / 'enkidu_alive'}")
    print(f"  Tracks:        {root / 'tracks'}")


if __name__ == "__main__":
    main()
