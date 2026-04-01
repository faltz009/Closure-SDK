"""Entry point for the Closure EA computer umbrella."""

import sys


def main() -> int:
    print("Closure EA")
    print("  dna: closure_ea.dna")
    print("  vm:  closure_ea/vm")
    print("  demo: closure_ea.enkidu_alive")
    print("Run DNA CLI with: python -m closure_ea.dna")
    return 0


if __name__ == "__main__":
    sys.exit(main())
