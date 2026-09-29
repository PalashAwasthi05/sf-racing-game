#!/bin/zsh
# Start the benchmark server (default port 3107). Needs Node 22+ on PATH; the viewer's own node_modules are linked in.
cd "$(dirname "$0")"
[ -e node_modules ] || ln -s ../king-street/node_modules node_modules
exec node ../king-street/node_modules/vite/bin/vite.js --config ./vite.config.mjs
