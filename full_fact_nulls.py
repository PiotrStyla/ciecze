#!/usr/bin/env python3
"""Secondary same-norm random-direction diagnostic added after discovery; see protocol."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from full_fact_flow import CASES, SEED, Model, aligned

DRAW_COUNT = 20


def sample(model, chunk, start, stop):
    if not (chunk['start'] <= start < stop <= chunk['stop']):
        raise ValueError('Null range must be contained in primary patch chunk')
    if chunk['layers'] != [2] or chunk['split'] != 'confirmation':
        raise ValueError('Wrong confirmation layer or split')
    patches = {(r['pair'], r['swapped'], r['target']):r for r in chunk['patches']}
    cases = json.loads(CASES.read_text(encoding='utf-8'))['confirmation']
    if len(patches) != 4 * (chunk['stop'] - chunk['start']):
        raise ValueError('Incomplete primary intervention chunk')
    rows = []
    for idx in range(start, stop):
        for swap in (0, 1):
            for target in (0, 1):
                patch = patches[idx, swap, target]
                texts, spans = aligned(model, cases[idx], swap, target)
                if patch['recipient'] != texts['recipient'] or patch['recipient_span'] != spans['recipient']:
                    raise ValueError('Recipient differs from primary intervention')
                positions = spans['recipient']
                receiver = model.states(texts['recipient'])[2][positions]
                sign = 1 if patch['expected'] == 'czerwony' else -1
                norm = patch['opposite_norm']
                for draw in range(DRAW_COUNT):
                    seed = SEED + idx * 10000 + swap * 1000 + target * 100 + draw
                    generator = torch.Generator(device='cpu').manual_seed(seed)
                    delta = torch.randn(receiver.shape, generator=generator, dtype=receiver.dtype)
                    delta *= norm / delta.norm()
                    null_margin = model.margin(texts['recipient'],
                                               patch=(2, [positions], [receiver + delta]))
                    rows.append(dict(pair=idx, swapped=swap, target=target, draw=draw,
                                     seed=seed, opposite_norm=norm, null_margin=null_margin,
                                     null_effect=sign * (patch['same_margin'] - null_margin)))
        print(f'null {idx + 1}/{len(cases)}: {cases[idx]}', flush=True)
    return rows


def analyze(chunks, result):
    rows = [r for chunk in chunks for r in chunk['rows']]
    indexed = {}
    for r in rows:
        key = (r['pair'], r['swapped'], r['target'], r['draw'])
        if key in indexed or not (0 <= key[0] < 16 and key[1] in (0, 1) and
                                  key[2] in (0, 1) and 0 <= key[3] < DRAW_COUNT):
            raise ValueError('Duplicate or out-of-range null row')
        indexed[key] = r
    if len(indexed) != 16 * 4 * DRAW_COUNT:
        raise ValueError('Incomplete null distribution')
    means = []
    for draw in range(DRAW_COUNT):
        by_pair = [np.mean([indexed[i, swap, target, draw]['null_effect']
                            for swap in (0, 1) for target in (0, 1)]) for i in range(16)]
        means.append(float(np.mean(by_pair)))
    donor = result['mean']
    exceed = sum(value >= donor for value in means)
    return dict(draws=DRAW_COUNT, donor_mean=donor, null_means=means,
                null_mean=float(np.mean(means)), null_sd=float(np.std(means, ddof=1)),
                null_min=float(np.min(means)), null_max=float(np.max(means)),
                exceed=exceed, p_emp=(1 + exceed) / (DRAW_COUNT + 1),
                stronger_description=(result['passes'] and exceed == 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--part', choices=('sample', 'analyze'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--primary', type=Path)
    ap.add_argument('--result', type=Path)
    ap.add_argument('--inputs', nargs='+', type=Path)
    ap.add_argument('--start', type=int)
    ap.add_argument('--stop', type=int)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if args.part == 'analyze':
        if not args.inputs or not args.result:
            ap.error('Analyze needs null chunks and primary result')
        result = analyze([json.loads(p.read_text(encoding='utf-8')) for p in args.inputs],
                         json.loads(args.result.read_text(encoding='utf-8')))
    else:
        if not args.primary or args.start is None or args.stop is None:
            ap.error('Sample needs primary patch chunk and start/stop')
        if not (0 <= args.start < args.stop <= 16):
            ap.error('Invalid null range')
        frozen = json.loads(CASES.read_text(encoding='utf-8'))
        primary = json.loads(args.primary.read_text(encoding='utf-8'))
        if primary['model'] != frozen['model'] or primary['revision'] != frozen['revision']:
            raise ValueError('Primary intervention model differs from frozen model')
        torch.set_num_threads(args.threads)
        model = Model(frozen['model'], revision=frozen['revision'])
        result = dict(start=args.start, stop=args.stop, model=frozen['model'],
                      revision=model.revision, layer=2,
                      rows=sample(model, primary, args.start, args.stop))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'part':args.part, 'rows':len(result.get('rows', [])),
                      'analysis':result if args.part == 'analyze' else None,
                      'output':str(args.out)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
