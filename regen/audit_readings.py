"""Batch missing-observation queue. Usage: audit_readings.py INPUT_DIR OUTPUT.json

Internal accounting only. This does not score image resemblance or invent data.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regen.reading_coverage import audit_reading


def audit_directory(root):
    records = []
    for path in sorted(Path(root).glob('*.json')):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or not raw.get('id'):
            continue
        records.append({'side': raw['id'], 'reading': str(path), **audit_reading(raw)})
    counts = Counter(issue['code'] for r in records for issue in r['issues'])
    return {'method': 'feature-accounting-v1', 'imageVerified': False,
            'summary': {'readings': len(records), 'unmodelledItems': sum(r['unmodelledCount'] for r in records),
                        'issueCounts': dict(counts)}, 'readings': records}


if __name__ == '__main__':
    report = audit_directory(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))
