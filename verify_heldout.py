#!/usr/bin/env python3
"""Recompute the frozen gate and paired bootstrap from published result records.

Integrity of file bytes is checked separately by verify_results.py's SHA-256 manifest.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/heldout'
MODELS = ('gate-gollem', 'gate-qwen05', 'gate-qwen15')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_baseline(result, pairs):
    require(result['pairs'] == pairs, 'Pair split differs from frozen list')
    rows = result['baseline']
    require(len(rows) == 4 * len(pairs), 'Incomplete two-colors-by-two-questions baseline')
    by_case = {}
    for row in rows:
        idx, swapped, target = row['pair'], row['swapped'], row['target']
        require(0 <= idx < len(pairs) and swapped in (0, 1) and target in (0, 1), 'Case index invalid')
        require(tuple(row['objects']) == tuple(pairs[idx]), 'Object identities changed')
        expected = ('czerwony', 'niebieski')[swapped ^ target]
        require(row['expected'] == expected, 'Expected color does not follow frozen assignment')
        require(row['prompt'].endswith(f'Jaki kolor ma {pairs[idx][target]}? Odpowiedz jednym słowem:'),
                'Question target changed')
        require(math.isfinite(row['margin']), 'Invalid answer log odds')
        require(row['forced'] == ('czerwony' if row['margin'] > 0 else 'niebieski'),
                'Forced choice differs from score')
        require(row['forced_ok'] == (row['forced'] == expected), 'Forced answer flag inconsistent')
        cleaned = row['free'].strip().strip('"\'„”').strip()
        match = re.fullmatch(r'(czerwony|niebieski)[.!?,;:]?', cleaned, re.IGNORECASE)
        parsed = match.group(1).lower() if match else None
        require(row['free_parsed'] == parsed and row['free_ok'] == (parsed == expected),
                'Free answer parser flag inconsistent')
        require((idx, swapped, target) not in by_case, 'Duplicate baseline question')
        by_case[(idx, swapped, target)] = row
    return by_case


def quality(by_case, count):
    forced = sum(all(by_case[(idx, swapped, target)]['forced_ok'] for target in (0, 1))
                 for idx in range(count) for swapped in (0, 1))
    free = sum(all(by_case[(idx, swapped, target)]['free_ok'] for target in (0, 1))
               for idx in range(count) for swapped in (0, 1))
    return forced, free


def main():
    frozen = load(ROOT / 'heldout_cases.json')
    gate, heldout = frozen['gate'], frozen['heldout']
    require(len(gate) == 12 and len(heldout) == 20, 'Wrong split sizes')
    nouns = [word for pair in gate + heldout for word in pair]
    require(len(set(nouns)) == len(nouns), 'Training and heldout object overlap')
    gates = {}
    for name in MODELS:
        result = load(DATA / f'{name}.json')
        require(result['split'] == 'gate', 'A model gate used the heldout split')
        by_case = verify_baseline(result, gate)
        forced, free = quality(by_case, len(gate))
        passed = forced >= 18 and free >= 18
        require(result['gate'] == dict(contexts=24, forced_both=forced, free_both=free, passes=passed),
                'Recorded model gate differs from raw questions')
        gates[name] = result
        print(f'{name}: forced {forced}/24, free {free}/24, passes {passed}')
    require(not gates['gate-gollem']['gate']['passes'] and
            not gates['gate-qwen05']['gate']['passes'] and
            gates['gate-qwen15']['gate']['passes'], 'Model order or selection differs from recorded plan')
    result = load(DATA / 'selected.json')
    require(result['split'] == 'heldout' and result['model'] == gates['gate-qwen15']['model'] and
            result['revision'] == gates['gate-qwen15']['revision'], 'Selected model/checkpoint mismatches gate')
    prior = load(DATA / 'baseline-qwen15.json')
    require(prior['model'] == result['model'] and prior['revision'] == result['revision'] and
            prior['baseline'] == result['baseline'], 'Intervention baseline differs from previously recorded heldout read')
    by_case = verify_baseline(result, heldout)
    forced, free = quality(by_case, len(heldout))
    require(result['heldout_quality'] == dict(contexts=40, forced_both=forced, free_both=free),
            'Recorded heldout quality differs from raw answers')
    print(f'heldout: forced {forced}/40, free {free}/40')
    rows = result['patches']
    require(len(rows) == 20 * 2 * 2 * 2, 'Incomplete paired intervention grid')
    seen, pair_effects = set(), []
    for idx in range(20):
        effects = []
        for swapped in (0, 1):
            for target in (0, 1):
                for layer in (5, 10):
                    matches = [r for r in rows if (r['pair'], r['swapped'], r['target'], r['layer']) ==
                               (idx, swapped, target, layer)]
                    require(len(matches) == 1, 'Missing or duplicated intervention')
                    row = matches[0]
                    key = (idx, swapped, target, layer)
                    require(key not in seen, 'Duplicate intervention')
                    seen.add(key)
                    require(row['objects'] == heldout[idx] and row['correct_color'] == by_case[(idx, swapped, target)]['expected'],
                            'Intervention fact assignment differs from baseline')
                    require(row['recipient'] == by_case[(idx, swapped, target)]['prompt'],
                            'Intervention recipient differs from baseline prompt')
                    a, b = heldout[idx]
                    ca, cb = (('czerwony', 'niebieski') if swapped == 0 else
                              ('niebieski', 'czerwony'))
                    suffix = f'Jaki kolor ma {heldout[idx][target]}? Odpowiedz jednym słowem:'
                    require(row['recipient'] == f'{a.capitalize()} jest {ca}. {b.capitalize()} jest {cb}. {suffix}' and
                            row['same'] == f'{a.capitalize()} ma kolor {ca}. {b.capitalize()} ma kolor {cb}. {suffix}' and
                            row['opposite'] == f'{a.capitalize()} ma kolor {cb}. {b.capitalize()} ma kolor {ca}. {suffix}',
                            'Intervention fact content or target differs from frozen task')
                    require(len(row['recipient_positions']) == len(row['same_positions']) ==
                            len(row['opposite_positions']) == 2 and
                            all(len(s) == 1 for key in ('recipient_positions', 'same_positions', 'opposite_positions')
                                for s in row[key]), 'Fact end positions not one token each')
                    require(all(math.isfinite(row[key]) for key in ('base_margin', 'self_margin',
                        'same_margin', 'opposite_margin', 'same_norm', 'opposite_norm', 'effect')),
                        'Non-finite intervention result')
                    require(abs(row['self_margin'] - row['base_margin']) <= 1e-4 and
                            abs(row['base_margin'] - by_case[(idx, swapped, target)]['margin']) <= 1e-4,
                            'Self-patch or original margin changed')
                    direction = 1 if row['correct_color'] == 'czerwony' else -1
                    require(math.isclose(row['effect'], direction * (row['same_margin'] - row['opposite_margin']),
                                         abs_tol=1e-7), 'Paired content effect inconsistent')
                    if layer == 5:
                        effects.append(row['effect'])
        pair_effects.append(float(np.mean(effects)))
    recorded = result['analysis']
    require(np.allclose(recorded['pair_effects'], pair_effects, rtol=0, atol=1e-6),
            'Pair aggregation inconsistent')
    rng = np.random.default_rng(frozen['seed'])
    samples = rng.choice(np.array(pair_effects), size=(10000, 20), replace=True).mean(axis=1)
    interval = np.percentile(samples, [2.5, 97.5])
    positive = sum(v > 0 for v in pair_effects)
    meets = interval[0] > 0 and positive >= 15 and forced >= 30 and free >= 30
    require(math.isclose(recorded['mean_effect'], float(np.mean(pair_effects)), abs_tol=1e-7) and
            np.allclose(recorded['bootstrap_95'], interval, rtol=0, atol=1e-6) and
            recorded['positive_pairs'] == positive and recorded['meets_rule'] == meets,
            'Bootstrap or decision rule inconsistent')
    print(f'interventions: {len(rows)} rows, pair mean {np.mean(pair_effects):+.4f}, '
          f'95% interval [{interval[0]:+.4f}, {interval[1]:+.4f}], positive {positive}/20; rule {meets}')


if __name__ == '__main__':
    main()
