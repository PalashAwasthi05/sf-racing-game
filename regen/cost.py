"""Token and dollar cost of this regeneration, from the Claude Code transcripts (every API call's usage by model) and
the Jev call log. Prices are list prices per million tokens; cache writes at 1.25x input (5 min) and reads at 0.1x."""
import argparse, glob, json, os, pathlib, collections
PRICE = {  # $/Mtok: input, output, cache read (list prices, claude-api reference cached 2026-09-25); cache writes 1.25x input
    'claude-haiku-4-5': (1.0, 5.0, .10), 'claude-sonnet-5-5': (2.0, 10.0, .20), 'claude-opus-5-5': (4.0, 20.0, .20)}
def family(m):
    for k in PRICE:
        if m and m.startswith(k): return k
    return m
def usage_of(path, since=None):
    tot = collections.defaultdict(lambda: collections.Counter())
    for line in open(path):
        try: e = json.loads(line)
        except Exception: continue
        msg = e.get('message') or {}
        u = msg.get('usage'); m = msg.get('model')
        if not u or not m or m == '<synthetic>': continue
        if since and e.get('timestamp', '') < since: continue
        c = tot[family(m)]
        c['calls'] += 1; c['input'] += u.get('input_tokens', 0); c['output'] += u.get('output_tokens', 0)
        c['cache_write'] += u.get('cache_creation_input_tokens', 0); c['cache_read'] += u.get('cache_read_input_tokens', 0)
    return tot
def dollars(m, c):
    pi, po, pr = PRICE.get(m, (0, 0, 0))
    return (c['input'] * pi + c['cache_write'] * pi * 1.25 + c['cache_read'] * pr + c['output'] * po) / 1e6
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--agents', nargs='*', default=[]); ap.add_argument('--since'); a = ap.parse_args()
    D = pathlib.Path(os.path.expanduser('~/.claude/projects/-Users-palash-Documents-reflex-code/ddc00244-2c8c-468c-aea6-4b87447bf172'))
    for ag in a.agents:
        f = D / 'subagents' / f'agent-{ag}.jsonl'
        for m, c in usage_of(f).items():
            print(ag, m, dict(c), '$%.3f' % dollars(m, c))
