#!/usr/bin/env python3
"""Check recorded whole-fact grids and recompute selection/interval, without model inference."""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HERE = ROOT / 'results/full-fact'
LAYERS = [2, 5, 10, 15, 20]


def load(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def check(condition, description):
    if not condition:
        raise ValueError(description)


def check_baseline(record, pairs, model, revision):
    check(record['model'] == model and record['revision'] == revision and record['pairs'] == pairs,
          'Wrong model, revision or frozen noun pairs in baseline')
    rows = record['baseline']
    found = {}
    for row in rows:
        key = row['pair'], row['swapped'], row['target']
        check(key not in found and 0 <= key[0] < len(pairs) and key[1] in (0, 1) and key[2] in (0, 1),
              'Duplicate or out-of-range baseline row')
        a, b = pairs[key[0]]
        colors = ('czerwony', 'niebieski') if key[1] == 0 else ('niebieski', 'czerwony')
        fact = f'{a.capitalize()} jest {colors[0]}. {b.capitalize()} jest {colors[1]}.'
        expected_prompt = f'Opis: {fact} Jaki kolor ma {pairs[key[0]][key[2]]}? Odpowiedz jednym słowem:'
        check(row['prompt'] == expected_prompt and row['expected'] == colors[key[2]],
              'Baseline task input differs from frozen plan')
        check(row['forced_ok'] == ((row['margin'] > 0) == (row['expected'] == 'czerwony')) and
              row['free_ok'] == (row['free_parsed'] == row['expected']), 'Baseline correctness flag is wrong')
        check(math.isfinite(row['margin']), 'Nonfinite baseline margin')
        found[key] = row
    check(len(found) == 4 * len(pairs), 'Missing baseline rows')
    counts = {'contexts': 2 * len(pairs), 'forced_both': 0, 'free_both': 0, 'joint_both': 0}
    for i in range(len(pairs)):
        for swap in (0, 1):
            first, second = found[i, swap, 0], found[i, swap, 1]
            f = first['forced_ok'] and second['forced_ok']
            g = first['free_ok'] and second['free_ok']
            counts['forced_both'] += int(f)
            counts['free_both'] += int(g)
            counts['joint_both'] += int(f and g)
    check(counts == record['quality'], 'Baseline quality differs from raw rows')
    return found, counts


def check_chunks(chunks, split, pairs, model, revision, baseline, expected_layers):
    found = {}
    controls = []
    for chunk in chunks:
        check(chunk['split'] == split and chunk['model'] == model and chunk['revision'] == revision and
              chunk['layers'] == expected_layers, 'Wrong split/model/layers in intervention chunk')
        check(0 <= chunk['start'] < chunk['stop'] <= len(pairs), 'Invalid chunk boundary')
        controls.extend(chunk['controls'])
        for row in chunk['patches']:
            key = row['pair'], row['swapped'], row['target'], row['layer']
            check(key not in found and chunk['start'] <= key[0] < chunk['stop'] and
                  key[1] in (0, 1) and key[2] in (0, 1) and key[3] in expected_layers,
                  'Duplicate, out-of-range or wrong-layer patch')
            b = baseline[key[:3]]
            check(row['objects'] == pairs[key[0]] and row['expected'] == b['expected'] and
                  row['recipient'] == b['prompt'] and row['base_margin'] == b['margin'],
                  'Patch uses wrong recipient or baseline')
            a, c = pairs[key[0]]
            recipient_colors = ('czerwony', 'niebieski') if key[1] == 0 else ('niebieski', 'czerwony')
            opposite_colors = recipient_colors[::-1]
            def donor(colors):
                return (f'Dane: {a.capitalize()} jest {colors[0]}. {c.capitalize()} jest {colors[1]}. '
                        f'Jaki kolor ma {pairs[key[0]][key[2]]}? Odpowiedz jednym słowem:')
            check(row['same'] == donor(recipient_colors) and row['opposite'] == donor(opposite_colors),
                  'Donor semantic content differs from frozen contrast')
            spans = [row[f'{name}_span'] for name in ('recipient', 'same', 'opposite')]
            check(all(span and span == list(range(span[0], span[-1] + 1)) for span in spans) and
                  len({span[0] for span in spans}) == 1 and len({len(span) for span in spans}) == 1,
                  'Unaligned fact-token spans in recorded intervention')
            for field in ('base_margin', 'same_margin', 'opposite_margin', 'same_norm', 'opposite_norm', 'effect'):
                check(math.isfinite(row[field]), f'Nonfinite {field}')
            check(row['same_norm'] >= 0 and row['opposite_norm'] >= 0, 'Negative norm')
            sign = 1 if b['expected'] == 'czerwony' else -1
            check(math.isclose(row['effect'], sign * (row['same_margin'] - row['opposite_margin']),
                               abs_tol=1e-7), 'Paired effect sign or magnitude differs from raw margins')
            found[key] = row
    check(len(found) == len(pairs) * 4 * len(expected_layers), 'Incomplete intervention grid')
    return found, controls


def pair_effects(found, layer, n):
    return [float(np.mean([found[i, swap, target, layer]['effect']
                           for swap in (0, 1) for target in (0, 1)])) for i in range(n)]


def main():
    frozen = json.loads((ROOT / 'full_fact_cases.json').read_text(encoding='utf-8'))
    model, revision = frozen['model'], frozen['revision']
    check(len(set(sum(frozen['discovery'] + frozen['confirmation'], []))) == 48,
          'Frozen noun pairs overlap')
    discovery = load('discovery-baseline.json')
    db, dq = check_baseline(discovery, frozen['discovery'], model, revision)
    confirm_baseline = load('confirmation-baseline.json')
    cb, cq = check_baseline(confirm_baseline, frozen['confirmation'], model, revision)
    dchunks = [load('discovery-a.json'), load('discovery-b.json')]
    dfound, dcontrols = check_chunks(dchunks, 'discovery', frozen['discovery'], model, revision,
                                    db, LAYERS)
    readouts = [c for c in dcontrols if c['kind'] == 'readout']
    selfs = [c for c in dcontrols if c['kind'] == 'self']
    check(len(readouts) == 1 and len(selfs) == 5 and
          all(c['error'] <= 1e-4 for c in selfs) and
          readouts[0]['max_logit_error'] <= 1e-4 and
          readouts[0]['max_donor_recipient_difference'] > 1e-4,
          'Discovery readout or self-patch control failed')
    selection = load('selection.json')
    means = {str(layer): float(np.mean(pair_effects(dfound, layer, 8))) for layer in LAYERS}
    selected = max(LAYERS, key=lambda layer: (means[str(layer)], -layer))
    check(selection['quality'] == dq and selection['mean_by_layer'] == means and
          selection['selected'] == selected and dq['joint_both'] >= 12,
          'Discovery selection does not follow preregistered rule')
    cchunks = [load('confirmation-a.json'), load('confirmation-b.json')]
    cfound, ccontrols = check_chunks(cchunks, 'confirmation', frozen['confirmation'], model,
                                    revision, cb, [selected])
    check(len(ccontrols) == 64 and all(c['kind'] == 'self' and c['error'] <= 1e-4
                                      for c in ccontrols), 'Confirmation self-patch controls incomplete')
    effects = pair_effects(cfound, selected, 16)
    rng = np.random.default_rng(frozen['seed'])
    samples = rng.choice(np.array(effects), size=(10000, 16), replace=True).mean(axis=1)
    low, high = (float(x) for x in np.percentile(samples, [2.5, 97.5]))
    result = load('result.json')
    positives = sum(effect > 0 for effect in effects)
    decision = cq['joint_both'] >= 24 and positives >= 12 and low > 0
    check(result['layer'] == selected and result['quality'] == cq and
          np.allclose(result['pair_effects'], effects, atol=1e-7, rtol=0) and
          math.isclose(result['mean'], np.mean(effects), abs_tol=1e-7) and
          np.allclose(result['bootstrap_95'], [low, high], atol=1e-7, rtol=0) and
          result['positive_pairs'] == positives and result['passes'] == decision and
          result['readout_control'] == readouts[0],
          'Confirmation effect, interval or decision differs from raw data')
    null_records = [load(f'null-{a}-{a + 4}.json') for a in (0, 4, 8, 12)]
    nulls = {}
    for a, chunk in zip((0, 4, 8, 12), null_records):
        check(chunk['start'] == a and chunk['stop'] == a + 4 and chunk['layer'] == selected and
              chunk['model'] == model and chunk['revision'] == revision, 'Invalid null chunk provenance')
        for row in chunk['rows']:
            key = row['pair'], row['swapped'], row['target'], row['draw']
            check(key not in nulls and a <= key[0] < a + 4 and key[1] in (0, 1) and
                  key[2] in (0, 1) and 0 <= key[3] < 20, 'Duplicate or invalid null cell')
            original = cfound[key[0], key[1], key[2], selected]
            seed = frozen['seed'] + key[0] * 10000 + key[1] * 1000 + key[2] * 100 + key[3]
            sign = 1 if original['expected'] == 'czerwony' else -1
            check(row['seed'] == seed and row['opposite_norm'] == original['opposite_norm'] and
                  math.isfinite(row['null_margin']) and
                  math.isclose(row['null_effect'], sign * (original['same_margin'] - row['null_margin']),
                               abs_tol=1e-7), 'Null seed, norm or contrast differs from primary data')
            nulls[key] = row
    check(len(nulls) == 16 * 4 * 20, 'Incomplete matched-norm null grid')
    null_means = []
    for draw in range(20):
        null_means.append(float(np.mean([
            np.mean([nulls[i, swap, target, draw]['null_effect']
                     for swap in (0, 1) for target in (0, 1)]) for i in range(16)])))
    secondary = load('null-analysis.json')
    exceed = sum(value >= result['mean'] for value in null_means)
    check(secondary['draws'] == 20 and secondary['donor_mean'] == result['mean'] and
          np.allclose(secondary['null_means'], null_means, atol=1e-7, rtol=0) and
          secondary['exceed'] == exceed and secondary['p_emp'] == (1 + exceed) / 21 and
          secondary['stronger_description'] == (decision and exceed == 0) and
          math.isclose(secondary['null_mean'], float(np.mean(null_means)), abs_tol=1e-7) and
          math.isclose(secondary['null_sd'], float(np.std(null_means, ddof=1)), abs_tol=1e-7) and
          math.isclose(secondary['null_min'], min(null_means), abs_tol=1e-7) and
          math.isclose(secondary['null_max'], max(null_means), abs_tol=1e-7),
          'Matched-norm null distribution differs from raw null cells')
    print(f'Matched-norm null: 20 draws, mean {secondary["null_mean"]:+.4f}, '
          f'exceed {exceed}/20, empirical p {secondary["p_emp"]:.4f}')
    print(f'Full fact: discovery quality {dq["joint_both"]}/16, selected layer {selected}; '
          f'confirmation quality {cq["joint_both"]}/32, pairs {positives}/16, '
          f'mean {result["mean"]:+.4f}, 95% interval [{low:+.4f}, {high:+.4f}], '
          f'preregistered rule {decision}; 160 discovery and 64 confirmation patch rows')


if __name__ == '__main__':
    main()
