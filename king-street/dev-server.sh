#!/bin/zsh
# Local preview server for the King Street world. The vinext launcher is a shell wrapper that needs
# `node` on PATH; the project has no system Node, so the Codex runtime is prepended here.
cd "$(dirname "$0")"
export PATH="/Users/palash/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:/Users/palash/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback:$PATH"
# Bind IPv4 directly; no IPv4-to-IPv6 bridge is needed.
exec ./node_modules/.bin/vinext dev --hostname 127.0.0.1 --port 3001
