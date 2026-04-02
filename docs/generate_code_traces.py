from __future__ import annotations

import io
import json
import tokenize
from pathlib import Path

import closure_rs


SNIPPETS = {
    'piece3': {
        'algorithm': 'source_code_embedding',
        'title': 'Multiply Function',
        'language': 'python',
        'source': '''def multiply(a, b):\n    result = a * b\n    return result\n''',
    },
    'piece4': {
        'algorithm': 'source_code_embedding',
        'title': 'Numerical Integration',
        'language': 'python',
        'source': '''def integrate(f, a, b, n=256):\n    dx = (b - a) / n\n    total = 0.0\n    for i in range(n):\n        x = a + (i + 0.5) * dx\n        total += f(x) * dx\n    return total\n''',
    },
}


def tokens_for(source: str) -> list[bytes]:
    out: list[bytes] = []
    stream = io.StringIO(source)
    for tok in tokenize.generate_tokens(stream.readline):
        if tok.type in {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        }:
            continue
        if not tok.string.strip() and tok.type != tokenize.OP:
            continue
        out.append(tok.string.encode('utf-8'))
    return out


def build_trace(piece: str, meta: dict) -> dict:
    records = tokens_for(meta['source'])
    path = closure_rs.path_from_raw_bytes('Sphere', records, hashed=False)
    states = [path.running_product(i).tolist() for i in range(len(records) + 1)]
    sigmas = [float(path.check_range(0, i)) if i > 0 else 0.0 for i in range(len(records) + 1)]
    branches = ['Start'] + ['Embedded'] * len(records)
    return {
        'algorithm': meta['algorithm'],
        'piece': piece,
        'title': meta['title'],
        'language': meta['language'],
        'source': meta['source'],
        'records': [r.decode('utf-8') for r in records],
        'states': states,
        'sigmas': sigmas,
        'branches': branches,
        'count': len(records),
    }


def main() -> None:
    docs = Path(__file__).resolve().parent
    for piece, meta in SNIPPETS.items():
        trace = build_trace(piece, meta)
        out = docs / f'seeing-code-trace-{piece}.json'
        out.write_text(json.dumps(trace, indent=2))
        print(out.name, trace['count'])


if __name__ == '__main__':
    main()
