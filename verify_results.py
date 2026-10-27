#!/usr/bin/env python3
"""Validate published artifacts and recompute descriptive counts; no model required.

Run: python verify_results.py
SHA-256 covers the published files, not the manifest itself. This checks recorded
invariants and data integrity, not statistical validity or new model inference.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_manifest():
    manifest = load_json(ROOT / 'reproducibility/manifest.json')
    for entry in manifest['files']:
        path = (ROOT / entry['path']).resolve()
        require(path.is_relative_to(ROOT), f'Unsafe manifest path: {entry["path"]}')
        data = path.read_bytes()
        require(len(data) == entry['bytes'], f'Size mismatch: {entry["path"]}')
        require(hashlib.sha256(data).hexdigest() == entry['sha256'], f'Hash mismatch: {entry["path"]}')
    print(f'SHA-256: {len(manifest["files"])} published files verified')


def verify_patches(folder, experiment):
    summary = load_json(folder / 'summary.json')
    with (folder / 'patches.csv').open(encoding='utf-8', newline='') as f:
        patches = list(csv.DictReader(f))
    with (folder / 'traces.csv').open(encoding='utf-8', newline='') as f:
        traces = list(csv.DictReader(f))
    groups = {}
    by_case = {b['case']: b for b in summary['baselines']}
    max_state = max(int(r['state']) for r in traces)
    for trace in traces:
        b = by_case[trace['case']]
        ids = b['prompt_ids'][0]
        if experiment == 'valve':
            require(len(ids) == len(b['prompt_ids'][1]), 'Unaligned valve prompts')
            changed = [i for i, (a, other) in enumerate(zip(ids, b['prompt_ids'][1])) if a != other]
            require(changed == b['regions']['question_subject'], 'Unexpected changed valve positions')
            if int(trace['token']) < min(changed):
                require(abs(float(trace['sensitivity'])) <= 1e-5, 'Retroactive recorded sensitivity')
        require(0 <= int(trace['token']) < len(ids), 'Trace token outside prompt')
        require(math.isfinite(float(trace['sensitivity'])), 'Nonfinite sensitivity')
    for row in patches:
        require(math.isfinite(float(row['patched_margin'])), 'Nonfinite intervention outcome')
        key = tuple(row[k] for k in ['case', 'direction', 'layer', 'region'])
        group = groups.setdefault(key, {})
        require(row['kind'] not in group, f'Duplicate intervention: {key}')
        group[row['kind']] = row
        if experiment == 'valve' and row['region'] == 'facts_control':
            require(abs(float(row['perturbation_norm'])) < 1e-5, 'Changed identical fact state')
            require(abs(float(row['patched_margin']) - float(row['recipient_margin'])) < 1e-4,
                    'Identical fact patch changed output')
    for key, pair in groups.items():
        require(set(pair) == {'donor', 'norm_matched_null'}, f'Missing paired control: {key}')
        require(math.isclose(float(pair['donor']['perturbation_norm']),
                             float(pair['norm_matched_null']['perturbation_norm']),
                             rel_tol=1e-4, abs_tol=1e-8), f'Unmatched recorded norms: {key}')
    expected_patches = sum(len(b['regions']) for b in by_case.values()) * len(summary['sampled_blocks']) * 2 * 2
    if experiment == 'dye':
        expected_patches *= 2  # two independently changed spans
    require(len(patches) == expected_patches, 'Incomplete intervention grid')
    expected_traces = sum(len(b['prompt_ids'][0]) for b in by_case.values()) * (max_state + 1)
    if experiment == 'dye':
        expected_traces *= 2
    require(len(traces) == expected_traces, 'Incomplete trace grid')
    for b in by_case.values():
        require(b['max_self_patch_error'] <= 1e-4, 'Self-patch control failed')
    if experiment == 'valve':
        bs = list(by_case.values())
        correct = sum(sum(b['correct_preferences']) for b in bs)
        paired = sum(all(b['correct_preferences']) for b in bs)
        red = sum(m > 0 for b in bs for m in b['margins_red_minus_blue'])
        for b in bs:
            signs = [-1, 1] if b['swapped'] else [1, -1]
            require([m * sign > 0 for m, sign in zip(b['margins_red_minus_blue'], signs)]
                    == b['correct_preferences'], 'Incorrect preference classification')
        analysis = load_json(folder / 'analysis.json')
        require(analysis['correct_question_preferences'] == correct, 'Analysis question count mismatch')
        require(analysis['correct_pairs'] == paired, 'Analysis pair count mismatch')
        require(analysis['red_preferences'] == red, 'Analysis color count mismatch')
        result = f'{correct}/{2 * len(bs)} individual preferences; {paired}/{len(bs)} pairs; red {red}/{2 * len(bs)}'
    else:
        passed = sum(b['red_preferred'] and b['blue_preferred'] for b in by_case.values())
        for b in by_case.values():
            m = b['margins_red_minus_blue']
            require((m[0] > 0) == b['red_preferred'] and (m[1] < 0) == b['blue_preferred'],
                    'Incorrect dye preference classification')
        result = f'{passed}/{len(by_case)} pairs; distractor lowers red margin in ' + str(
            sum(b['distractor_shift'] < 0 for b in by_case.values())) + f'/{len(by_case)} cases'
    print(f'{folder.relative_to(ROOT)}: {len(traces)} trace rows, {len(patches)} interventions; {result}')


def verify_bernoulli():
    with (ROOT / 'results/bernoulli/layers.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        require(all(math.isfinite(float(v)) for v in row.values()), 'Nonfinite Bernoulli diagnostic')
    def cv(values):
        average = sum(values) / len(values)
        return (sum((v - average) ** 2 for v in values) / len(values)) ** .5 / average
    print('results/bernoulli: ' + str(len(rows)) + ' rows; ' + ', '.join(
        f'CV({name})={cv([float(r[name]) for r in rows]):.3f}' for name in ['v_times_A', 'v', 'A_pr']))


def verify_binding(folder):
    summary = load_json(folder / 'summary.json')
    cases = {b['case']: b for b in summary['cases']}
    def rows(name):
        with (folder / name).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    baselines = rows('baselines.csv')
    patches = rows('patches.csv')
    transfer = rows('transfers.csv')
    grouped_margins = {}
    for row in baselines:
        margin = float(row['margin_red_minus_blue'])
        require(math.isfinite(margin), 'Nonfinite binding baseline')
        key = (row['case'], row['condition'], row['question'])
        require(key not in grouped_margins, 'Duplicate binding baseline')
        grouped_margins[key] = margin
        target = 0 if row['question'] == 'key' else 1
        require(row['prompt'] == cases[row['case']]['prompts'][row['condition']][target],
                'Binding baseline prompt mismatch')
        if row['condition'] == 'N':
            require(row['correct_preference'] == '', 'Neutral context has no known correct color')
    for b in cases.values():
        m = {c: [grouped_margins[(b['case'], c, q)] for q in ['key', 'cup']] for c in ['A', 'B', 'N']}
        key, cup = m['A'][0] - m['B'][0], m['A'][1] - m['B'][1]
        require(math.isclose((key - cup) / 2, b['binding_interaction'], abs_tol=1e-6),
                'Binding interaction differs from baseline CSV')
        require(math.isclose((key + cup) / 2, b['common_context_effect'], abs_tol=1e-6),
                'Common context effect differs from baseline CSV')
        for target in range(2):
            a = b['prompt_ids']['A'][target]
            for condition in ['B', 'N']:
                other = b['prompt_ids'][condition][target]
                require(len(a) == len(other), 'Unaligned binding contexts')
                require([i for i, (x, y) in enumerate(zip(a, other)) if x != y]
                        == b['regions']['fact_colors'], 'Unexpected changed binding positions')
        require(b['max_self_patch_error'] <= 1e-4, 'Binding self-patch control failed')
        require(b['max_null_norm_relative_error'] <= 1e-4, 'Binding norm control failed')
        require(b['max_earlier_relative_difference'] <= 1e-5, 'Binding causality control failed')
    paired, shifted = {}, {}
    for row in patches:
        key = tuple(row[k] for k in ['case', 'direction', 'layer', 'region', 'kind'])
        target = row['question']
        group = shifted.setdefault(key, {})
        require(target not in group, 'Duplicate binding intervention')
        actual_shift = float(row['patched_margin']) - float(row['recipient_margin'])
        require(math.isfinite(actual_shift), 'Nonfinite binding intervention')
        require(math.isclose(actual_shift, float(row['margin_shift']), abs_tol=1e-6),
                'Binding shift is inconsistent with recorded margins')
        group[target] = actual_shift
        norm_key = key[:-1] + (target,)
        pair = paired.setdefault(norm_key, {})
        pair[row['kind']] = float(row['perturbation_norm'])
    for pair in paired.values():
        require(set(pair) == {'donor', 'norm_matched_null'}, 'Missing binding null control')
        require(math.isclose(pair['donor'], pair['norm_matched_null'], rel_tol=1e-4, abs_tol=1e-8),
                'Binding norm mismatch in recorded interventions')
    seen = set()
    for row in transfer:
        key = tuple(row[k] for k in ['case', 'direction', 'layer', 'region', 'kind'])
        require(key not in seen and key in shifted, 'Duplicate or orphan binding transfer')
        seen.add(key)
        shifts = shifted[key]
        require(set(shifts) == {'key', 'cup'}, 'Incomplete paired-question transfer')
        sign = 1 if row['direction'] == 'A_to_B' else -1
        selective = sign * (shifts['key'] - shifts['cup']) / 2
        common = sign * (shifts['key'] + shifts['cup']) / 2
        require(math.isclose(selective, float(row['selective_shift']), abs_tol=1e-6),
                'Selective transfer is inconsistent with paired interventions')
        require(math.isclose(common, float(row['common_shift']), abs_tol=1e-6),
                'Common transfer is inconsistent with paired interventions')
    expected = len(cases) * 2 * 2 * len(summary['sampled_blocks']) * 3 * 2
    require(len(baselines) == len(cases) * 6, 'Incomplete context/question baseline grid')
    require(len(patches) == expected and len(transfer) * 2 == expected, 'Incomplete binding grid')
    require(seen == set(shifted), 'Missing derived binding transfers')
    print(f'{folder.relative_to(ROOT)}: {len(patches)} interventions; positive interaction '
          f'{sum(b["binding_interaction"] > 0 for b in cases.values())}/{len(cases)} cases')


def verify_replicate(folder):
    summary = load_json(folder / 'summary.json')
    cases_path = folder / 'cases.csv'
    cases = []
    if cases_path.exists():
        with cases_path.open(encoding='utf-8', newline='') as f:
            cases = list(csv.DictReader(f))
        require(len(cases) == len(summary['cases']), 'Case CSV is incomplete')
    by_case = {c['case']: c for c in summary['cases']}
    if cases:
        require(set(by_case) == {c['case'] for c in cases}, 'Case identifiers differ between CSV and JSON')
    for case, c in by_case.items():
        key = c['margins']['A'][0] - c['margins']['B'][0]
        cup = c['margins']['A'][1] - c['margins']['B'][1]
        require(math.isclose((key - cup) / 2, c['binding_interaction'], abs_tol=1e-6),
                f'Interaction inconsistent with margins: {case}')
        require(math.isclose((key + cup) / 2, c['common_context_effect'], abs_tol=1e-6),
                f'Common effect inconsistent with margins: {case}')
        lengths = {len(ids) for target in range(2) for ids in
                   (c['prompt_ids'][condition][target] for condition in ['A', 'B', 'N'])}
        require(len(lengths) == 1, f'Unaligned contexts survived validation: {case}')
        for target in range(2):
            base = c['prompt_ids']['A'][target]
            for condition in ['B', 'N']:
                other = c['prompt_ids'][condition][target]
                require([i for i, (x, y) in enumerate(zip(base, other)) if x != y]
                        == c['regions']['fact_colors'], f'Unexpected change in {condition}: {case}')
    for rejection in summary['rejected']:
        require(rejection['reason'] and rejection['case'] not in by_case, 'Invalid rejection record')
    if (folder / 'traces.csv').exists():
        with (folder / 'traces.csv').open(encoding='utf-8', newline='') as f:
            traces = list(csv.DictReader(f))
        max_state = 1 + max(int(t['state']) for t in traces)
        expected = sum(len(c['prompt_ids']['A'][0]) * 2 * 2 * max_state for c in by_case.values())
        require(len(traces) == expected, 'Replication trace grid is incomplete')
        for trace in traces:
            c = by_case[trace['case']]
            if int(trace['token']) < min(c['regions']['fact_colors']):
                require(abs(float(trace['sensitivity'])) <= 1e-5, 'Retroactive recorded sensitivity')
    if (folder / 'nulls.csv').exists():
        with (folder / 'nulls.csv').open(encoding='utf-8', newline='') as f:
            nulls = list(csv.DictReader(f))
        for row in nulls:
            draws = int(row['draws'])
            exceed = int(row['exceed'])
            require(0 <= exceed <= draws, 'Null exceed count out of range')
            require(math.isclose(float(row['p_empirical']), (1 + exceed) / (draws + 1), abs_tol=1e-9),
                    'Empirical p-value is inconsistent with exceedance count')
            require(float(row['null_min']) <= float(row['null_mean']) <= float(row['null_max']),
                    'Null summary statistics are not ordered')
        stay = sum(float(r['p_empirical']) <= 1 / (int(r['draws']) + 1) + 1e-12 for r in nulls)
        print(f'{folder.relative_to(ROOT)}: {len(cases)} cases, {len(summary["rejected"])} rejected, '
              f'{len(nulls)} null rows; donor beyond every null draw in {stay}/{len(nulls)}')
    else:
        positives = sum(c['binding_interaction'] > 0 for c in by_case.values())
        if by_case:
            print(f'{folder.relative_to(ROOT)}: {len(cases)} cases, {len(summary["rejected"])} rejected; '
                  f'positive interaction {positives}/{len(cases)}')



def main():
    verify_manifest()
    verify_bernoulli()
    for directory, experiment in [('dye', 'dye'), ('valve', 'valve'),
                                  ('smoke/dye', 'dye'), ('smoke/valve', 'valve')]:
        verify_patches(ROOT / 'results' / directory, experiment)
    verify_binding(ROOT / 'results/binding')
    verify_binding(ROOT / 'results/smoke/binding')
    verify_replicate(ROOT / 'results/replication')
    verify_replicate(ROOT / 'results/null-calibration')
    print('Recorded evidence checks passed. No new inference or statistical significance test was performed.')


if __name__ == '__main__':
    main()
