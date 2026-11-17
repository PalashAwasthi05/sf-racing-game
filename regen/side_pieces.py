"""Load an area's base and upper-tier side pieces without silent ID collisions."""
import json
from pathlib import Path


def load_side_pieces(sides_dir):
    """Return pieces in file order; identical repeated IDs are included only once.

    The base file is required. The tier file is optional, but conflicting records
    in either file are errors: choosing one would change the reading's frame.
    """
    sides_dir = Path(sides_dir)
    pieces, origins = {}, {}
    for name in ('sides.json', 'sides-tiers.json'):
        path = sides_dir / name
        if name == 'sides-tiers.json' and not path.exists():
            continue
        with path.open() as source:
            records = json.load(source)['pieces']
        for index, piece in enumerate(records):
            piece_id = piece['id']
            location = f'{path}:pieces[{index}]'
            if piece_id in pieces:
                if pieces[piece_id] != piece:
                    raise ValueError(f'Conflicting side piece ID {piece_id!r}: '
                                     f'{origins[piece_id]} and {location}')
                continue
            pieces[piece_id] = piece
            origins[piece_id] = location
    return list(pieces.values())
