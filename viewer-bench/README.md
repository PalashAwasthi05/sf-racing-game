# Viewer benchmark

Measures the King Street viewer (`work/king-street/app`) in headless Chromium on the machine's real GPU (ANGLE Metal),
off to the side of the running preview: its own Vite server (port 3107), its own copies of the viewer modules, the live
`public/` folder. Nothing here is part of the app, and editing files here never reloads the preview on :3001.

Why a harness: frame times read in the in-app browser pane are unreliable (the pane throttles `requestAnimationFrame`
when hidden), and `?viewFrom` screenshots cannot say what a change cost. Here the animation loop is stepped with fixed
timestamps, every frame is timed with and without `gl.finish()`, and `Math.random`, the GTAO noise and the capture clock
are pinned, so two captures of one pose differ only by the code under test.

## Use

```
./serve.sh                               # leave running; Node 22+ on PATH
./snapshot.sh live                       # copy app/*.ts into variants/live
node run.mjs live out-live --captures    # report.json + same-pose PNGs
node run.mjs pre-budget out-pre --captures
python3 pixdiff.py out-pre out-live      # per-pose pixel differences
```

`run.mjs` needs Playwright; `PLAYWRIGHT_FROM` names a directory whose `node_modules` has it (default: the SoMa repo).
Scenarios: the default street station, two street poses, two aerial poses, Drive idle and Drive accelerating (six
simulated seconds, so a reflection capture and far-cascade redraws are included). Times are CPU+GPU per frame with the
GPU drained each frame, so they are a cost, not a frame rate.

In the page, `__bench.frames(n, {keys, gpu})`, `__bench.callBreakdown()` (draws per render target),
`__bench.capture(w, h)` and `__bench.rawCapture(w, h, {passIndex: enabled})` are available; `__view.budget.flags`
switches the depth proxies and the transmission-copy culling live for A/B.

Noise floor: two runs of one build differ on up to ~0.1% of pixels (a few hundred outliers on the aerial poses).
GPU timer queries work but carry a floor of a few milliseconds per query on ANGLE Metal; attribute cost by switching
features off instead.

`variants/pre-budget` is the viewer as it was before the frame-budget pass of 2026-09-29 (NEXT_AGENT.md).

## Driving at 60 fps (2026-10-08)

`run.mjs`'s stepped, GPU-drained times understate what driving feels like: on this machine the frame is limited by the
browser's GPU process (it translates every WebGL call into Metal and sits at 100 % of a core), the GPU clocks down
between drained frames, and a 5K window renders 2.5x the pixels of the default viewport. Three tools measure it as used:

```
node drive.mjs <variant> <out> --w=2560 --h=1330 --dpr=2 --seconds=40 --free --hz=60     # the Studio Display (60 Hz)
node drive.mjs <variant> <out> --w=1512 --h=860 --dpr=2 --seconds=40 --free              # the laptop's screen (120 Hz)
node drive.mjs <variant> <out> --w=2560 --h=1330 --dpr=2 --free --uncapped               # no vsync: throughput
node drive.mjs <variant> <out> ... "--js=__view.quality(1.25, true)"                     # fixed render scale
node ab.mjs <variant> '{"name":["<js on>","<js off>"]}' [--scale=1.25]                    # live switch A/B
node profile.mjs <variant> <out.cpuprofile> --seconds=8                                  # CPU profile while driving
node heap.mjs <variant> <out-dir>                                                        # what the load allocates
```

`drive.mjs` drives the car round the review route with an autopilot (steering for a point 10 m ahead, ~40 km/h) and
records every frame (interval, the viewer's own JS time, draws, triangles, probe face, heap); `over20` (frames that
missed a 60 Hz deadline) is the number that matters. Headless Chromium paces `--free` at 120 Hz here. `ab.mjs`
alternates baseline and configuration at three street views in one page, so slow drifts cancel.

Traps:
- Variant names must match `[a-z0-9_-]`: `perfF` silently loads `variants/live` (an old copy). Use lowercase.
- A new variant directory needs `touch bench.ts` (its import list is built when bench.ts is transformed); files deleted
  and copied back into a variant are served stale until touched (`touch variants/<name>/*.ts`).
- Other sessions' jobs (SegFormer/MPS photo work, ring builds, viewer captures) and macOS's own media analysis
  share the GPU: check `top` and ask before a run; compare variants interleaved, never across a gap.
- GPU timer queries are unusable on ANGLE Metal (a 1x1 pass reads 16 ms); the JS time of a frame shows back-pressure
  (it rises toward the whole frame when the GPU or its process is behind), which the viewer's adaptive quality uses.
- `drive.mjs --trace=<file.json>` records a Chrome trace of every process (about 400 MB for 9 s). A long frame whose page
  JS is short shows in the GPU process as `IOSurfaceImageBacking::WaitForCommandsToBeScheduled` (the GPU was behind);
  `SharedImageStub::OnCreateSharedImage` next to it is a canvas resize. Tracing changes the timing: a hitch seen in
  most plain runs did not appear in three traced ones.
- Pace as the display does. The Studio Display (5K) runs at 60 Hz and the laptop's screen at 120 Hz
  (`system_profiler SPDisplaysDataType -json`); headless Chromium paces `--free` at 120 Hz, so a 5K run without
  `--hz=60` keeps the GPU at 100 % and turns the reflection probe's face renders into a backlog (50-250 ms frames at
  the sixth face). At 60 Hz four 5K drives in five missed no frame; one still had ~2 s of 25 ms frames after its first
  probe capture (cause not found). With `--hz=60` an interval of 20-22 ms is pacing jitter; a missed frame is 25 ms or
  more.
- The load leaves ~1 GB of garbage (heap ~1.6 GB, ~0.6 GB live; `heap.mjs` lists the allocators). It is not what
  stalls the probe: `--gc` (a full collection before the drive) left those frames as they were.
