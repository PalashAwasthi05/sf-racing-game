"""Three-way merge independent source-reading revisions without dropping fields.

Readers can re-observe different feature classes from the same images. A newer
file is not automatically a complete replacement: merge against their common
base and fail on competing edits. Inputs remain immutable and hashed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


class MergeConflict(ValueError):
    pass


_MISSING = object()


def merge_value(base, branches, path='$'):
    edits = []
    for value in branches:
        if value != base and not any(value == old for old in edits):
            edits.append(value)
    if not edits:
        return base
    if len(edits) == 1:
        return edits[0]
    if isinstance(base, dict) and all(isinstance(v, dict) for v in edits):
        out = {}
        for key in sorted(set(base).union(*(v.keys() for v in edits))):
            result = merge_value(base.get(key, _MISSING), [v.get(key, _MISSING) for v in edits], f'{path}.{key}')
            if result is not _MISSING:
                out[key] = result
        return out
    if isinstance(base, list) and all(isinstance(v, list) and len(v) == len(base) for v in edits):
        return [merge_value(value, [v[i] for v in edits], f'{path}[{i}]') for i, value in enumerate(base)]
    raise MergeConflict(f'competing source-reading changes at {path}')


def read_directory(directory):
    readings = {}
    for path in sorted(Path(directory).glob('*.json')):
        doc = json.loads(path.read_text())
        # Reports/coverage manifests live next to observations but are not readings.
        if not isinstance(doc, dict) or not isinstance(doc.get('id'), str):
            continue
        if doc['id'] in readings:
            raise ValueError(f'duplicate reading identity in {directory}: {doc["id"]}')
        readings[doc['id']] = (doc, path)
    return readings


def merge_directories(base_dir, branch_dirs, output):
    base = read_directory(base_dir)
    branches = [read_directory(d) for d in branch_dirs]
    identities = sorted(set(base).union(*(b.keys() for b in branches)))
    merged = {}
    audit = []
    for identity in identities:
        original = base.get(identity, (_MISSING, None))[0]
        updates = [b[identity][0] if identity in b else original for b in branches]
        merged[identity] = merge_value(original, updates, identity)
        sources = [base[identity][1]] if identity in base else []
        sources += [b[identity][1] for b in branches if identity in b]
        audit.append({'id': identity, 'changed': merged[identity] != original,
                      'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]})
    # Resolve all conflicts before creating output; a failed merge cannot masquerade
    # as a complete generation input.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for identity, reading in merged.items():
        if '/' in identity or '\\' in identity or identity in ('.', '..'):
            raise ValueError('reading identity must be a filename stem')
        (output / f'{identity}.json').write_text(json.dumps(reading, indent=2) + '\n')
    report = {'base': str(base_dir), 'branches': list(map(str, branch_dirs)), 'readings': audit,
              'count': len(merged), 'changed': sum(r['changed'] for r in audit)}
    (output / 'merge-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--branch', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = merge_directories(args.base, args.branch, args.output)
    print(json.dumps({k: report[k] for k in ('count', 'changed')}))


if __name__ == '__main__':
    main()
