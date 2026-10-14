# SF racing game

A street-level reconstruction of San Francisco built from public map data, LiDAR and street panoramas, plus a
browser viewer you can drive around in. The pilot area is the few blocks around 188 King Street; Pacific Heights is
the second area.

The repository has two parts:

| Folder | What it is |
| --- | --- |
| `./` (root) | The Python world pipeline: fetches inputs, generates buildings, streets and materials, checks the result and publishes it to the viewer. |
| `king-street/` | The three.js viewer (vinext/Vite + React): orbit, street-level walk, plan view and drive mode. |

Only source and small config are in git. Generated worlds (`*.glb`, `*.bin`), downloaded imagery, LiDAR, model
weights and build folders (`clean-v*/`, `detailed-v*/`, ...) are ignored; the full working folder is about 100 GB.
A fresh clone therefore has no world to show until you build and publish one (or copy the published chunks over
from a machine that has them, see "Viewer" below).

## Viewer

Requirements: Node 22.13 or newer and pnpm.

```bash
cd king-street
pnpm install
pnpm exec vinext dev --hostname 127.0.0.1 --port 3001
```

Open http://127.0.0.1:3001/. `pnpm install` applies the patch in `patches/` to vinext (it fixes "Maximum call stack
size exceeded" 500s from the dev server). The port is strict: if 3001 is taken the server refuses to start rather
than moving to another port, so stop the old server first (`lsof -nP -iTCP:3001 -sTCP:LISTEN`).

`king-street/dev-server.sh` does the same thing on Palash's machine, where Node is not on PATH and the script
prepends a local runtime. Don't run `ipv4-bridge.py`; it is obsolete.

The viewer loads `public/model.json`, which names the world chunks (`public/world-<hash>-NN.bin`). Those chunks are
not in git: produce them with the publish step below, or copy `king-street/public/*.bin` and `*.glb` from a machine
that has them.

Useful URLs:

- `/?manifest=/compare/<name>.json` loads another build or area (the area menu in the top bar does this too).
- `/car-studio?model=<url of a .glb>` is the review page for the player's car model.

Controls:

- **Orbit:** drag to orbit, scroll to zoom, right-drag to pan.
- **Street:** W A S D to move (Shift to go faster), drag to look.
- **Drive** (car button): W / S throttle and brake, A / D steer, Space handbrake, C camera, R reset, drag to look.
- **Tour** plays a route through the area.

## Pipeline

Requirements: Python 3.12 and macOS or Linux (the shell scripts are zsh). The pipeline expects its virtualenv at
`geoenv/` and is run from the folder above this repository's root with `PYTHONPATH=work`, so the checkout should
be named `work`.

```bash
python3.12 -m venv geoenv
geoenv/bin/pip install -r general-world-test-requirements.txt
```

That covers generation and the tests. Add `imagery-requirements.txt` for the panorama tools and
`vision-requirements.txt` (torch, transformers) for the photo readers and facade judge. The judges load their model
weights offline from a Hugging Face cache in `.models/` (SegFormer `nvidia/segformer-b2-finetuned-ade-512-512` and
Grounding DINO `IDEA-Research/grounding-dino-tiny`); download them there once, e.g. with
`huggingface-cli download <model> --cache-dir .models`.

### Inputs

Inputs are fetched into local folders that git ignores:

- `area_fetch.py <area dir> --lat LAT --lon LON` fetches one area's map inputs into `<area dir>/data`: DataSF
  datasets, OpenStreetMap and the USGS 2023 LiDAR. The area folder needs its own `config.json` (anchor and radii);
  copy one from `areas/pacific-heights/`.
- `fetch_panoramas_v32.py` fetches street panoramas into `imagery-street-v12/`. It needs a Mapillary access token,
  read from stdin; never commit it.
- `fetch_horizon.py` fetches the terrain and water used for the viewer's horizon into `data/horizon`.

### Build, check, publish

Run these from the folder that contains `work/`. Heavy steps use a lot of memory; run one at a time.

```bash
zsh work/run_buildings_v22.sh work/clean-vNN
```

Builds the pilot world into `work/clean-vNN` (generator, height audit, textures, materials). It does not publish.

```bash
zsh work/areas/build_area.sh pacific-heights <label>
```

Builds a whole area end to end under `work/areas/<area>/<label>` and writes a preview manifest the viewer can open
with `?manifest=`.

Before publishing a pilot build, render the views that have been flagged in review and look at every sheet:

```bash
PYTHONPATH=work work/geoenv/bin/python work/check_flagged_views.py --candidate work/clean-vNN
```

```bash
PYTHONPATH=work work/geoenv/bin/python work/judge_facades_v27.py --help
```

compares facades against their photographs. Then publish into the viewer:

```bash
PYTHONPATH=work work/geoenv/bin/python work/publish_full_candidate_v24.py --candidate work/clean-vNN --label "0.NN / NAME"
```

The publish script refuses when flagged-view sheets are missing or stale, when a flagged building's wall colour no
longer matches its photograph, when a material drops to a placeholder, or when walls no photograph covers changed
without a cause. Its `--help` lists the waiver flags; use them only with a recorded reason.

### Tests

From the folder that contains `work/`:

```bash
PYTHONPATH=work work/geoenv/bin/python -m pytest work/test_*.py
```

Run the root `test_*.py` files only. A plain `pytest` also collects the copies under
`king-street/pipeline-source/` and fails.

## More documentation

- `pipeline/README.md` and `general_world/README.md`: the generator's modules and rules.
- `viewer-bench/README.md`: frame-time benchmarks and A/B comparisons for the viewer.
- `king-street/README.md`: notes from the 2026-09-08 pilot (historical).
- Current state, decisions and agent rules live outside the repository in `../agent-handoff/` (`START_HERE.md`
  first).
