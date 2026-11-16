#!/bin/zsh
set -eu
cd "${0:A:h}"
play_runtime="${PLAYWRIGHT_FROM:-$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node}"
if [[ -x "$play_runtime/bin/node" ]]; then
  export PATH="$play_runtime/bin:$PATH"
fi
if node -e 'process.exit(typeof WebSocket === "function" ? 0 : 1)' 2>/dev/null; then
  export STREAM_DIRECT_CDP=1
elif [[ ! -d "$play_runtime/node_modules/playwright" ]]; then
  echo "This drive needs Node with built-in WebSocket support or the local Playwright runtime."
  exit 1
fi
reports="$(mktemp -d "${TMPDIR:-/tmp}/sf-manual-drive.XXXXXX")"
echo "Drive for up to two minutes. W accelerates, S brakes, A/D steer. Reports: $reports"
export PLAYWRIGHT_FROM="$play_runtime" STREAM_MANUAL=1 STREAM_WIDTH="${STREAM_WIDTH:-1920}" STREAM_SECONDS=120
export STREAM_NEIGHBORHOOD="${STREAM_NEIGHBORHOOD:-1}"
exec python3 ../viewer-bench/guard_memory.py --mib 2816 --reserve-mib 3584 --timeout 155 \
  --report "$reports/memory.json" --merge-report "$reports/drive.json" \
  node ../viewer-bench/stream-performance.mjs . "$reports/drive.json"
