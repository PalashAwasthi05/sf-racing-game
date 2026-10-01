"""Minimal Jev (TypeSafe System One) client: one request per state with several questions, every call's usage logged
for the cost report. The key is read from ~/.typesafe-key and never printed."""
import json, os, pathlib, time, urllib.request
KEY = pathlib.Path(os.path.expanduser('~/.typesafe-key')).read_text().strip()
URL = 'https://api.typesafe.ai/v1/systemone'
LOG = pathlib.Path(__file__).resolve().parent / 'v1' / 'jev-usage.jsonl'
PRICE_PER_MTOK = .042          # input only; output is free (docs.typesafe.ai/models)

def ask(state, questions, tag='', model='jev-latest', retries=4):
    body = json.dumps({'model': model, 'state': state, 'questions': questions}).encode()
    for k in range(retries):
        req = urllib.request.Request(URL, data=body, headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=60) as r: res = json.loads(r.read())
            u = res.get('usage', {})
            with open(LOG, 'a') as f: f.write(json.dumps({'t': time.time(), 'tag': tag, 'input': u.get('input_tokens', 0), 'output': u.get('output_tokens', 0), 'questions': len(questions)}) + '\n')
            return res['answers']
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503) and k < retries - 1: time.sleep(2 ** k); continue
            raise RuntimeError(f'jev {e.code}: {e.read()[:300]}')

def cost():
    rows = [json.loads(l) for l in open(LOG)] if LOG.exists() else []
    tin = sum(r['input'] for r in rows); tout = sum(r['output'] for r in rows)
    return {'calls': len(rows), 'inputTokens': tin, 'outputTokens': tout, 'usd': round(tin * PRICE_PER_MTOK / 1e6, 5)}
