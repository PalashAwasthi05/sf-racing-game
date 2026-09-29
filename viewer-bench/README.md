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
