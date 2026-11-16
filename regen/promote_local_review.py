"""Point the local area menu and current.json at a complete, checked world."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import tempfile
import zlib


def replace_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write((json.dumps(value, indent=1) + '\n').encode())
            handle.flush()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def promote(build, public):
    receipt = json.loads((build / 'review-build.json').read_text())
    if not receipt['fullWorldRebuilt'] or receipt.get('usesPerBuildingMassingOverrides'):
        raise ValueError('Expected a complete general-rule build')
    label = receipt['label']
    compare = public / 'compare'
    manifest = json.loads((compare / f'{label}.json').read_text())
    sidecars = {key: json.loads((public / manifest[key].lstrip('/')).read_text())
                for key in ('world', 'cars', 'overhead')}
    world = sidecars['world']
    if world['previewRevision'] != label or len(world['buildings']) != receipt['buildingCount']:
        raise ValueError('World identity or building count differs from the build')
    unzip = zlib.decompressobj(16 + zlib.MAX_WBITS)
    digest = hashlib.sha256()
    compressed = plain = 0
    for url in manifest['chunks']:
        with (public / url.lstrip('/')).open('rb') as chunk:
            while data := chunk.read(256 * 1024):
                block = unzip.decompress(data)
                compressed += len(data)
                plain += len(block)
                digest.update(block)
    if not unzip.eof or unzip.unused_data or plain != manifest['uncompressedBytes'] or compressed != manifest['compressedBytes']:
        raise ValueError('Model chunks are incomplete')
    if digest.hexdigest()[:12] != manifest['contentHash']:
        raise ValueError('Model content differs from the manifest')
    menu = json.loads((compare / 'areas.json').read_text())
    area = build.parent.name
    entry = next((row for row in menu['areas'] if row['id'] == area), None)
    if entry is None:
        entry = {'id': area, 'label': area.replace('-', ' ').title()}
        menu['areas'].append(entry)
    entry.update(detail=f"{world['label']} · {label}", manifest=f'/compare/{label}.json')
    replace_json(compare / 'current.json', manifest)
    replace_json(compare / 'areas.json', menu)
    print(f"{label} · {receipt['buildingCount']} buildings · {len(manifest['chunks'])} verified model chunks")


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('build', type=Path)
    args = cli.parse_args()
    promote(args.build.resolve(), Path(__file__).resolve().parents[1] / 'king-street/public')
