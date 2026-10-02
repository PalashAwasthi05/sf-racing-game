"""Receives viewer captures from the local preview page (POST /save?name=<file>.jpg) into <out dir> (default
work/areas/captures), for reviewing a build when the in-app browser pane is hidden. Listens on 127.0.0.1:3199 only and
accepts requests from the local viewer origin only. Page side (in the viewer's console or a review script):
  const b=await __view.capture(1400,860); fetch('http://127.0.0.1:3199/save?name=x.jpg',{method:'POST',body:b})
Usage: capture_receiver.py [out dir]"""
import http.server, pathlib, re, sys, urllib.parse
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent / 'captures'); OUT.mkdir(parents=True, exist_ok=True)
ORIGIN = 'http://127.0.0.1:3001'
class H(http.server.BaseHTTPRequestHandler):
    def cors(self):
        self.send_header('Access-Control-Allow-Origin', ORIGIN); self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    def do_OPTIONS(self):
        self.send_response(204); self.cors(); self.end_headers()
    def do_POST(self):
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query); name = (q.get('name') or ['shot.jpg'])[0]
        if self.headers.get('Origin') not in (None, ORIGIN) or not re.fullmatch(r'[\w.-]{1,80}\.(jpg|png)', name):
            self.send_response(403); self.end_headers(); return
        n = int(self.headers.get('Content-Length', 0)); (OUT / name).write_bytes(self.rfile.read(n))
        self.send_response(200); self.cors(); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(('127.0.0.1', 3199), H).serve_forever()
