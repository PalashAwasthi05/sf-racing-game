"""Account for observed facade features without inventing missing geometry.

This checks a reading's internal completeness, not whether its claims match an
image. Unknowns remain jobs for the reader. No place or building identities
select a rule.
"""

CATEGORIES = {
    'roofline': {'roofs', 'parapetTop', 'bands'},
    'surface_zones': {'materials'},
    'windows': {'banks', 'openings', 'bays'},
    'entrance': {'openings'},
    'garage': {'openings'},
    'projections': {'bays', 'porches', 'balconies', 'canopies', 'recesses'},
    'trim': {'bands', 'piers', 'quoins', 'ornaments', 'rustication', 'timbering'},
    'access': {'porches', 'openings', 'streetEdge'},
    'signage': {'signs'},
}
VISIBILITY = {'visible', 'absent', 'occluded', 'out_of_frame'}


def audit_reading(reading):
    issues = []

    def issue(code, path, detail):
        issues.append({'code': code, 'path': path, 'detail': detail})

    inventory = reading.get('inventory')
    if not isinstance(inventory, list) or not inventory:
        inventory = []
        issue('inventory_missing', 'inventory', 'No structured feature inventory; completeness is unverified.')
    seen = set()
    for i, item in enumerate(inventory):
        path = f'inventory[{i}]'
        if not isinstance(item, dict) or item.get('category') not in CATEGORIES:
            issue('inventory_invalid', path, 'Unknown feature category.'); continue
        cat = item['category']; seen.add(cat)
        visibility = item.get('visibility')
        if visibility not in VISIBILITY:
            issue('visibility_unknown', path, 'Record visible, absent, occluded, or out_of_frame.'); continue
        if not isinstance(item.get('sourceImages'), list) or not item['sourceImages']:
            issue('source_missing', path, 'Name the images inspected for this visibility claim.')
        targets = item.get('targets') or []
        unresolved = item.get('unmodelled')
        unresolved_ok = (isinstance(unresolved, int) and not isinstance(unresolved, bool)
                         and 0 <= unresolved < len(reading.get('unmodelled') or []))
        if visibility == 'visible' and not targets and not unresolved_ok:
            issue('observed_feature_unrepresented', path, 'Visible feature has neither geometry/material targets nor an unresolved record.')
        if visibility in ('absent', 'out_of_frame') and targets:
            issue('contradictory_visibility', path, 'An absent or out-of-frame feature cannot support geometry.')
        if visibility == 'occluded' and not item.get('reason'):
            issue('occlusion_reason_missing', path, 'Describe the obstruction; do not infer an absent feature.')
        for target in targets:
            if not isinstance(target, dict):
                issue('target_invalid', path, 'Target must identify a collection and index.'); continue
            collection, index = target.get('collection'), target.get('index')
            rows = reading.get(collection) if isinstance(collection, str) else None
            valid = (collection in CATEGORIES[cat] and isinstance(rows, list)
                     and isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(rows))
            if not valid:
                issue('target_missing', path, 'The claimed feature target is missing or belongs to another category.'); continue
            obj = rows[index]
            kind = obj.get('kind', 'window') if isinstance(obj, dict) else None
            if ((cat == 'entrance' and kind != 'entry') or
                (cat == 'garage' and kind not in ('garage', 'rollup')) or
                (cat == 'windows' and collection in ('banks', 'openings') and kind not in ('window', 'curtain', 'louvre', 'shopfront'))):
                issue('target_semantics_mismatch', path, 'The target does not represent this feature type.')
    for cat in sorted(set(CATEGORIES) - seen):
        issue('category_unreviewed', 'inventory.' + cat, 'No visibility/accounting decision for this category.')

    for i, mat in enumerate(reading.get('materials') or []):
        if isinstance(mat, dict) and mat.get('family') in ('timber', 'tile', 'terracotta', 'metal-panel', 'brick', 'stone', 'concrete') and not mat.get('pattern'):
            issue('surface_structure_unread', f'materials[{i}]', 'Family and color do not identify joints, board direction, bond, or shingle shape.')
    entries = [o for o in reading.get('openings') or [] if isinstance(o, dict) and o.get('kind') == 'entry']
    for i, opening in enumerate(reading.get('openings') or []):
        if isinstance(opening, dict) and opening.get('kind') == 'entry' and not isinstance(opening.get('door'), dict):
            issue('door_structure_unread', f'openings[{i}]', 'Entry opening has no door observation; a frame is not a door leaf.')
    if reading.get('porches') and not entries:
        # A porch is a cue to review access, never evidence for placing a door.
        entrance_known = any(isinstance(x, dict) and x.get('category') == 'entrance' and x.get('visibility') in ('absent', 'occluded', 'out_of_frame') for x in inventory)
        if not entrance_known:
            issue('porch_access_unread', 'porches', 'Porch present without an entry or an explicit entrance visibility decision. Reinspect the source.')

    unresolved = []
    for i, item in enumerate(reading.get('unmodelled') or []):
        unresolved.append({'kind': 'source-unmodelled', 'path': f'unmodelled[{i}]', 'observation': item})
    # A partial reading must remain visibly partial, including after compilation.
    return {'version': 1, 'status': 'needs_reading' if issues or unresolved else 'accounted_not_image_verified',
            'imageVerified': False, 'issues': issues, 'unresolved': unresolved,
            'reviewedCategories': sorted(seen), 'unmodelledCount': len(unresolved)}


def merge_coverage(*reports):
    """Retain pre-clean omissions; sanitizing a malformed record is not a fix."""
    reports = [r for r in reports if isinstance(r, dict)]
    if not reports: return audit_reading({})
    out = dict(reports[-1]); issues = {}; unresolved = {}
    for report in reports:
        for item in report.get('issues', []): issues[(item['code'], item['path'])] = item
        for item in report.get('unresolved', []): unresolved[item['path']] = item
    out.update(issues=list(issues.values()), unresolved=list(unresolved.values()),
               unmodelledCount=len(unresolved), imageVerified=False)
    out['status'] = 'needs_reading' if issues or unresolved else 'accounted_not_image_verified'
    return out
