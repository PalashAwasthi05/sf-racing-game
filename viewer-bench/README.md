# Viewer benchmark

## Full neighborhood at native 1080p

Run the built viewer on loopback. The circuit runner uses the installed Chrome executable in windowless mode through its debugging protocol. It verifies 60 Hz on an empty tab before loading the full map, then locks scene resolution at 1920 × 1080. The fixture steers the normal driving physics through four turns.

```sh
python3 guard_memory.py --mib 2816 --reserve-mib 3584 --timeout 135 \
  --report /tmp/circuit-memory.json node neighborhood-circuit.mjs \
  'http://127.0.0.1:3001/?drive=1&fps=1' \
  ../../agent-handoff/reviews/2026-11-12/full-map-1080/route.json /tmp/circuit.json
```

The report includes untrimmed animation intervals, distance, maximum deviation from the route, tile readiness, and the exact scene and output resolution check for every frame. The browser closes automatically. The watchdog covers the browser and driver. The already-running static server is outside its process tree.

Set `SF_CIRCUIT_NATIVE_DEFAULT=1` to exercise normal player settings and still check native resolution every frame. Set `SF_CIRCUIT_PRESENT=1` for compositor reports. Add `SF_CIRCUIT_PRESENT_MODE=display` to measure displayed intervals with an 8 MiB trace. Use `SF_CIRCUIT_GPU=1` for sampled GPU and CPU timings by pass and geometry memory snapshots before and after driving. ANGLE Metal timer queries can include queue delays. Use the presentation trace to evaluate delivered frames. The fixture checks the 20 crossing IDs in sequence. Partial reports preserve progress when a cutoff stops the drive.

On a busy Mac, add `--wait-for-green 3600` before `--report`. Only the small watchdog runs while waiting. Eight consecutive seconds of normal pressure and the configured reserve must be available before the browser launches. The normal memory limits apply immediately after launch.

## Memory-limited block drive on macOS

Build the local client first with `npm run build:local` in `../king-street`. Set `PLAYWRIGHT_FROM` to a local project containing Playwright and optionally set `CHROME_EXECUTABLE` to a Chrome binary. From this directory run:

```sh
BLOCK_WIDTH=1920 python3 guard_memory.py --mib 2816 --reserve-mib 3584 --timeout 75 --report /tmp/world-memory.json node block-performance.mjs ../king-street /tmp/world-drive.json
```

The benchmark serves the prebuilt game on a temporary loopback port, opens a private Chrome process and checks its requested model chunks against the scoped block manifest. The watchdog sums macOS physical footprints for the test tree including the server and browser helpers. It stops the job on its workload limit, warning memory pressure, low free-plus-inactive pages or timeout. Sampled usage can overshoot between polls. `BLOCK_ENTRY=comparison` verifies the baseline comparison page and rejects optional photo/splat downloads. The launcher is for isolated benchmarks. It cannot enforce a limit on the user's existing browser.

## Guarded streaming drive

Set `STREAM_RIDE_REVIEW=1` on a guarded drive to record body pitch and roll, wheel contact height differences, and the rear deck’s projected pixel position on each displayed frame. Compare at the same native resolution, route and input schedule.

Automated drives run in a windowless browser and leave desktop focus alone. On macOS the runner prefers the installed Chromium headless shell. A manually launched interactive drive opens a window for the player. Set `STREAM_VERIFY_60=1` to check 120 animation intervals in the same browser before loading the game. `STREAM_DIRECT_CDP=1` uses Node's built-in WebSocket to control the isolated browser and retains the asset allowlist and memory watchdog. For a separate visual run, `STREAM_FLASH_REVIEW=1` records canvas and compositor samples, regional dark patches with their neighboring frames, and image pairs across detail changes. `STREAM_ADAPTIVE=1` leaves automatic quality enabled. Frame readback can affect timing.

The daylight chase check also counts sustained black frames on the car and samples the scene color before and after post-processing on detection. Set `STREAM_FLASH_BASELINE` to a prior visual report to capture the corrected street at the driving poses of its saved vehicle blackouts.

For a two-minute manual 720p session, double-click `../king-street/Drive with memory cutoff.command` in Finder. It launches a fresh local Chrome window only if the memory preflight passes. It writes the drive and memory traces to a new temporary folder. Manual sessions may take any route. For the repeatable automatic benchmark, prepare the local client and `public/compare/drive-stream.json`, then run:

```sh
PLAYWRIGHT_FROM=/path/to/project-with-playwright python3 guard_memory.py --mib 2816 --reserve-mib 3584 --timeout 170 --report /tmp/stream-memory.json --merge-report /tmp/stream-drive.json node stream-performance.mjs ../king-street /tmp/stream-drive.json
```

This opens the real game at `/?stream=corridor&drive=1` in an isolated Chrome instance at native 1920×1080. It follows `routeEN` with throttle and steering keyboard events through the normal car physics, braking at the far endpoint and reversing heading in place only once stationary (below 0.8 km/h), so the same cells can be revisited with the same physics. Each heading reversal is explicitly logged as `stopped-turnaround`; position is preserved. It never steps the simulation or disables parked cars. The default drive lasts 110 seconds; `STREAM_SECONDS=10..125` changes that limit. A stalled car ends the run early and insufficient crossings fail validation. `STREAM_MANUAL=1` opens a visible isolated browser and leaves all controls to the user for the same measured interval. `CHROME_EXECUTABLE` can select a Chrome binary. `BUNDLE_DIR` selects an absolute path to an isolated prebuilt bundle instead of the project’s `.local-viewer`. Do not run this alongside other GPU or memory workloads.

`STREAM_NEIGHBORHOOD=1` selects `/compare/neighborhood-stream.json` and the full spatial neighborhood URL; the default benchmark remains the original four-block corridor. The Finder memory-cutoff launcher defaults to the neighborhood (`STREAM_NEIGHBORHOOD=0` selects the corridor).

`STREAM_WIDTH=1280` selects a clearly labelled native 1280×720 diagnostic run; the default remains 1920×1080. `STREAM_CAPTURE_FINAL=/absolute/path/final.jpg` saves a JPEG at quality 75 after timing ends.

The manifest must have `version:1`, `scope:"drive-stream"`, an east/north route, and at least four cells. When `pieces` is present, detailed downloads come exclusively from those packages, each with a `parentId` pointing to one semantic block in `cells`; the parent package chunks are blocked. `crossing.id` remains the semantic destination parent and `crossing.piece` identifies its ready resident entry piece. `prepare`/`attached`/`evicted` and `active` refer to pieces. Samples retain semantic `current` and physical `piece` separately. A `base` scene manifest supplies shared road/collision `contentHash` and `chunks`, which are also allowlisted and required to download. Each cell supplies `id`, `boundsEN`, `contentHash`, byte counts, and `/world-<contentHash>-...` chunk paths; no cell may exceed 150 MiB uncompressed. The benchmark blocks world chunks outside the union of those paths and rejects network/page/console errors. The shared base and each cell may supply an optional `.gpu` package with `profile:"s3tc-v1"`, `portableHash` matching the portable `contentHash`, its own `contentHash`, `chunks`, format and byte counts. Both profiles are allowlisted, but the first request locks that package to one encoding across eviction and return; mixed requests fail. Exactly one encoding, with all its chunks, must finish for the base and every cell actually attached. `packageEncodings` records that proof; prefetch requests are retained in the report.

The report contains every live frame interval, draw/triangle counts, path and renderer memory samples every 250 ms, stream events, and frame summaries within ±1 second of crossings versus steady driving. At least three crossings into three distinct destination cells must occur, with `ready:true` recorded on every crossing. At least one eviction must occur and no more than two detail cells may be active at once (shared base excluded; `STREAM_MAX_ACTIVE` overrides this limit). Each crossing includes its own FPS/p95/max summary and bracketing path samples. Preparation windows are measured from `prepare` to the matching `attached` event separately from actual boundary windows. The page must expose `__view.stream.status()` with `active`, `loading`, `ready`, and events: `crossing` events use `id` for the destination cell and `ms=performance.now()`, plus `positionEN` and destination `ready`. `__car.car.position()` supplies `[x,z,yaw]` (north is `-z`); `telemetry().speedKmh` and the live renderer debug interface are also required.

For external memory correlation, the watchdog preserves its existing compact sample rows and adds `startedEpochMs` and `sampleColumns`. A watchdog sample's wall time is `startedEpochMs + 1000 * sample[0]`. Drive samples have `epochMs`; stream event wall times are `timeOriginEpochMs + event.ms`. `launchEpochMs` and `guardStartedEpochMs` are also included. The memory file accounts for the complete isolated process tree, including Chrome/GPU helpers; in-page heap and renderer counters are supplementary. The drive report is checkpointed every second using only new frame, path and event records from the page; Node merges these into the safety snapshot. `--merge-report` merges external physical memory into it after completion or cutoff, including each crossing’s memory summary and early/middle/late settled windows after eviction. Settled windows exclude two seconds around streaming events and require ready, nonloading path samples. Empty windows explicitly report null statistics. `matchedVisitMemory` separately compares median footprint for the same sorted active piece set on outbound versus first return, requiring ready/nonloading samples and more than 400 ms since attach/eviction; each pair records sample counts and return-minus-outbound MiB. New traces include macOS pressure level as the fifth sample column and write a short report even when the initial headroom check prevents launch. The watchdog also records physical memory for individual owned processes and role totals (driver/browser/renderer/GPU/server/helpers) about once per second, and attaches role summaries to crossing memory. Only process IDs, roles and memory are retained; executable names, paths and command arguments are omitted. Always inspect the memory report’s `reason` and `exit` too.


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
