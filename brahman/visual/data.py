"""Grid walk data generation for Brahman Visual.

Generates closed walks (return to origin) and open walks (don't return)
on a 2D grid. Each walk is a sequence of action tokens.
"""

import random

# Action tokens
UP, DOWN, LEFT, RIGHT, EOS = 0, 1, 2, 3, 4
VOCAB_SIZE = 5
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT", EOS: "EOS"}

# Movement vectors
MOVES = {
    UP: (0, 1),
    DOWN: (0, -1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

INVERSES = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


def generate_closed_walk(min_steps=2, max_half=8):
    """Generate a walk that returns to origin.

    Takes N random steps, then computes the minimal return path
    (in shuffled order). The model learns to close from displacement,
    not by memorizing and reversing the forward sequence.
    """
    n = random.randint(min_steps, max_half)
    forward = [random.choice([UP, DOWN, LEFT, RIGHT]) for _ in range(n)]

    # Compute net displacement
    x, y = 0, 0
    for a in forward:
        dx, dy = MOVES[a]
        x, y = x + dx, y + dy

    # Build return: minimal steps to cancel displacement, shuffled
    backward = []
    if x > 0:
        backward.extend([LEFT] * x)
    elif x < 0:
        backward.extend([RIGHT] * (-x))
    if y > 0:
        backward.extend([DOWN] * y)
    elif y < 0:
        backward.extend([UP] * (-y))

    random.shuffle(backward)
    return forward + backward + [EOS]


def generate_open_walk(min_steps=4, max_steps=16):
    """Generate a walk that does NOT return to origin."""
    while True:
        n = random.randint(min_steps, max_steps)
        walk = [random.choice([UP, DOWN, LEFT, RIGHT]) for _ in range(n)]
        # Check it doesn't accidentally close
        x, y = 0, 0
        for a in walk:
            dx, dy = MOVES[a]
            x, y = x + dx, y + dy
        if x != 0 or y != 0:
            return walk + [EOS]


def walk_to_positions(walk):
    """Convert action sequence to list of (x, y) positions."""
    positions = [(0, 0)]
    x, y = 0, 0
    for a in walk:
        if a == EOS:
            break
        dx, dy = MOVES[a]
        x, y = x + dx, y + dy
        positions.append((x, y))
    return positions


def is_closed(walk):
    """Check if a walk returns to origin."""
    x, y = 0, 0
    for a in walk:
        if a == EOS:
            break
        dx, dy = MOVES[a]
        x, y = x + dx, y + dy
    return x == 0 and y == 0


def walk_length(walk):
    """Number of action tokens (excluding EOS)."""
    return sum(1 for a in walk if a != EOS)


def pad_batch(seqs):
    """Pad sequences to equal length. Returns (padded, lengths)."""
    max_len = max(len(s) for s in seqs)
    padded = []
    lengths = []
    for s in seqs:
        lengths.append(len(s))
        padded.append(s + [EOS] * (max_len - len(s)))
    return padded, lengths


def generate_dataset(n_closed, n_open=0):
    """Generate a dataset of closed (and optionally open) walks."""
    data = []
    for _ in range(n_closed):
        data.append(generate_closed_walk())
    for _ in range(n_open):
        data.append(generate_open_walk())
    return data


def walk_to_string(walk):
    """Pretty-print a walk."""
    arrows = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→", EOS: "·"}
    return "".join(arrows.get(a, "?") for a in walk)