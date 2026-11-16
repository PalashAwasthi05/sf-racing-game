"""Build after source changes, then replace the launcher with a small static server."""
import os
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parents[1]
os.chdir(project)
bundle = project / '.local-viewer'
index = bundle / 'index.html'
inputs = [project / name for name in ['package.json', 'pnpm-lock.yaml', 'tsconfig.json']]
for folder in ['local-viewer', 'app', 'components', 'hooks', 'lib']:
    inputs.extend(p for p in (project / folder).rglob('*') if p.suffix in ('.ts','.tsx','.css','.mjs','.html'))
if not index.exists() or any(p.stat().st_mtime_ns > index.stat().st_mtime_ns for p in inputs):
    subprocess.run(['node_modules/.bin/vite', 'build', '--config', 'local-viewer/vite.config.mjs'], check=True)
os.execv(sys.executable, [sys.executable, 'local-viewer/serve.py', '--bundle', str(bundle),
                          '--public', 'public', *sys.argv[1:]])
