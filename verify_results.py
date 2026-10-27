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


def main():
    verify_manifest()
    verify_bernoulli()
    for directory, experiment in [('dye', 'dye'), ('valve', 'valve'),
                                  ('smoke/dye', 'dye'), ('smoke/valve', 'valve')]:
        verify_patches(ROOT / 'results' / directory, experiment)
    print('Recorded evidence checks passed. No new inference or statistical significance test was performed.')


if __name__ == '__main__':
    main()
