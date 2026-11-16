"""Serve the built client on loopback and stream assets directly from public."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path
import argparse
from urllib.parse import urlsplit


class Viewer(SimpleHTTPRequestHandler):
    routes = {'/', '/previous', '/block-drive', '/car-studio'}

    def translate_path(self, request):
        path = urlsplit(request).path.rstrip('/') or '/'
        if path in self.routes:
            return str(self.server.bundle / 'index.html')
        safe = Path(super().translate_path(request)).relative_to(self.directory)
        root = self.server.bundle if safe.parts[:1] == ('assets',) and (self.server.bundle / safe).is_file() else Path(self.directory)
        file = (root / safe).resolve()
        return str(file if file.is_relative_to(root) else root / '.unavailable')

    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        path = urlsplit(self.path).path
        immutable = path.startswith('/world-') or path.startswith('/assets/') and Path(self.translate_path(self.path)).is_relative_to(self.server.bundle)
        self.send_header('Cache-Control', 'private, max-age=31536000, immutable' if immutable else 'no-cache')
        super().end_headers()

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--public', required=True, type=Path)
    parser.add_argument('--bundle', required=True, type=Path)
    args = parser.parse_args()
    if not (args.bundle / 'index.html').is_file():
        parser.error('Build the local viewer first')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Viewer, directory=str(args.public.resolve())))
    server.bundle = args.bundle.resolve()
    server.serve_forever()
