#!/usr/bin/env python3
"""Paired whole-fact patch experiment; plan and decision in docs/FULL_FACT_PROTOCOL.md."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from heldout_flow import Model, CANDIDATES, free_color

HERE = Path(__file__).resolve().parent
CASES = HERE / 'full_fact_cases.json'
LAYERS = (2, 5, 10, 15, 20)
SEED = 20261029


def colors(swapped):
    return CANDIDATES[swapped], CANDIDATES[1 - swapped]


def prompt(pair, swapped, target, label):
    a, b = pair
    ca, cb = colors(swapped)
    fact = f'{a.capitalize()} jest {ca}. {b.capitalize()} jest {cb}.'
    text = f'{label}: {fact} Jaki kolor ma {pair[target]}? Odpowiedz jednym słowem:'
    return text, fact


def fact_span(model, text, fact):
    formatted = model.prefix(text)
    ids, offsets = model.encode(formatted)
    beginning = formatted.find(text)
    if beginning < 0:
        raise ValueError('Question not found in chat prefix')
    start = formatted.find(fact, beginning)
    if start < 0 or formatted.find(fact, start + 1) >= 0:
        raise ValueError('Fact span must occur exactly once')
    end = start + len(fact)
    positions = [i for i, (left, right) in enumerate(offsets)
                 if right > start and left < end]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError('Fact span is empty or discontinuous')
    left = offsets[positions[0]][0]
    if (left < start and formatted[left:start] != ' ') or offsets[positions[-1]][1] > end:
        raise ValueError('Fact token merges with label or question')
    return positions


def aligned(model, pair, swapped, target):
    texts = {}
    spans = {}
    for kind, s, label in [('recipient', swapped, 'Opis'),
                           ('same', swapped, 'Dane'),
                           ('opposite', 1 - swapped, 'Dane')]:
        text, fact = prompt(pair, s, target, label)
        texts[kind] = text
        spans[kind] = fact_span(model, text, fact)
    counts = {k: len(v) for k, v in spans.items()}
    starts = {k: v[0] for k, v in spans.items()}
    if len(set(counts.values())) != 1 or len(set(starts.values())) != 1:
        raise ValueError(f'Full-fact alignment failure: {pair} {swapped} {target} {counts} {starts}')
    return texts, spans


def baseline(model, pairs):
    rows = []
    for idx, pair in enumerate(pairs):
        for swapped in (0, 1):
            for target in (0, 1):
                texts, spans = aligned(model, pair, swapped, target)
                text = texts['recipient']
                expected = colors(swapped)[target]
                margin = model.margin(text)
                free = model.free_answer(text)
                parsed = free_color(free)
                rows.append(dict(pair=idx, objects=pair, swapped=swapped, target=target,
                                 prompt=text, expected=expected, fact_span=spans['recipient'],
                                 margin=margin, free=free, free_parsed=parsed,
                                 forced_ok=(margin > 0) == (expected == CANDIDATES[0]),
                                 free_ok=parsed == expected))
        print(f'baseline {idx + 1}/{len(pairs)}: {pair}', flush=True)
    return rows


def quality(rows, n):
    by = {(row['pair'], row['swapped'], row['target']): row for row in rows}
    if len(by) != 4 * n or len(rows) != len(by):
        raise ValueError('Incomplete baseline grid')
    forced = free = joint = 0
    for idx in range(n):
        for swapped in (0, 1):
            pair_rows = [by[idx, swapped, target] for target in (0, 1)]
            f = all(row['forced_ok'] for row in pair_rows)
            g = all(row['free_ok'] for row in pair_rows)
            forced += f
            free += g
            joint += f and g
    return dict(contexts=2 * n, forced_both=forced, free_both=free, joint_both=joint)


def first_logit_control(model, recipient, opposite, source_states):
    recv_ids, _ = model.encode(model.prefix(recipient))
    donor_ids, _ = model.encode(model.prefix(opposite))
    last = len(model.blocks) - 1
    with torch.inference_mode():
        donor_logits = model.logits(donor_ids)[0, -1].float()
        receiver_logits = model.logits(recv_ids)[0, -1].float()
        handle = model.patch_hook(last, [[len(recv_ids) - 1]],
                                  [source_states[last][[len(donor_ids) - 1]]])
        try:
            patched_logits = model.logits(recv_ids)[0, -1].float()
        finally:
            handle.remove()
        difference = float((donor_logits - patched_logits).abs().max())
        contrast = float((donor_logits - receiver_logits).abs().max())
    if difference > 1e-4 or contrast <= 1e-4:
        raise AssertionError(f'Readout control fails: donor error={difference}; contrast={contrast}')
    return dict(max_logit_error=difference, max_donor_recipient_difference=contrast)


def patch_chunk(model, pairs, baseline_rows, start, stop, layers, discovery):
    index = {(row['pair'], row['swapped'], row['target']): row for row in baseline_rows}
    rows = []
    controls = []
    for idx in range(start, stop):
        pair = pairs[idx]
        for swapped in (0, 1):
            for target in (0, 1):
                texts, spans = aligned(model, pair, swapped, target)
                base = index[idx, swapped, target]
                if base['prompt'] != texts['recipient']:
                    raise ValueError('Saved baseline differs from intervention recipient')
                states = {kind: model.states(text) for kind, text in texts.items()}
                if discovery and idx == 0 and swapped == 0 and target == 0:
                    controls.append(dict(kind='readout',
                                         **first_logit_control(model, texts['recipient'],
                                                               texts['opposite'], states['opposite'])))
                sign = 1 if base['expected'] == CANDIDATES[0] else -1
                for layer in layers:
                    recv_positions = spans['recipient']
                    recv = states['recipient'][layer][recv_positions]
                    if not discovery or idx == 0 and swapped == 0 and target == 0:
                        self_margin = model.margin(texts['recipient'],
                                                   patch=(layer, [recv_positions], [recv]))
                        if abs(self_margin - base['margin']) > 1e-4:
                            raise AssertionError('Self-patch changed forced margin')
                        controls.append(dict(kind='self', pair=idx, swapped=swapped,
                                             target=target, layer=layer,
                                             error=abs(self_margin - base['margin'])))
                    values, norms = {}, {}
                    for kind in ('same', 'opposite'):
                        source = states[kind][layer][spans[kind]]
                        norms[kind] = float((source - recv).norm())
                        values[kind] = model.margin(texts['recipient'],
                                                    patch=(layer, [recv_positions], [source]))
                    rows.append(dict(pair=idx, objects=pair, swapped=swapped, target=target,
                                     layer=layer, recipient=texts['recipient'],
                                     same=texts['same'], opposite=texts['opposite'],
                                     recipient_span=recv_positions, same_span=spans['same'],
                                     opposite_span=spans['opposite'], expected=base['expected'],
                                     base_margin=base['margin'], same_margin=values['same'],
                                     opposite_margin=values['opposite'],
                                     same_norm=norms['same'], opposite_norm=norms['opposite'],
                                     effect=sign * (values['same'] - values['opposite'])))
        print(f'patch {idx + 1}/{len(pairs)}: {pair}', flush=True)
    return rows, controls


def per_pair(rows, layers, n):
    scores = {}
    for layer in layers:
        values = []
        for idx in range(n):
            effects = [r['effect'] for r in rows if r['layer'] == layer and r['pair'] == idx]
            if len(effects) != 4:
                raise ValueError(f'Incomplete paired effect: {idx} {layer}')
            values.append(float(np.mean(effects)))
        scores[layer] = values
    return scores


def select(chunks, baseline_record):
    rows = [r for c in chunks for r in c['patches']]
    controls = [c for chunk in chunks for c in chunk['controls']]
    scores = per_pair(rows, LAYERS, 8)
    gate = quality(baseline_record['baseline'], 8)
    if gate['joint_both'] < 12:
        return dict(quality=gate, selected=None, reason='Discovery quality below 12/16',
                    mean_by_layer={str(k):float(np.mean(v)) for k,v in scores.items()}, controls=controls)
    if not any(c['kind'] == 'readout' for c in controls) or len([c for c in controls if c['kind'] == 'self']) != 5:
        raise ValueError('Missing technical controls in discovery')
    means = {layer:float(np.mean(values)) for layer,values in scores.items()}
    chosen = max(LAYERS, key=lambda layer: (means[layer], -layer))
    return dict(quality=gate, selected=chosen,
                mean_by_layer={str(k):v for k,v in means.items()}, controls=controls)


def confirm(rows, baseline_record, layer):
    scores = per_pair(rows, [layer], 16)[layer]
    gate = quality(baseline_record['baseline'], 16)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(np.array(scores), size=(10000, len(scores)), replace=True).mean(axis=1)
    interval = [float(v) for v in np.percentile(samples, [2.5, 97.5])]
    positive = sum(value > 0 for value in scores)
    return dict(layer=layer, quality=gate, pair_effects=scores,
                mean=float(np.mean(scores)), bootstrap_95=interval,
                positive_pairs=positive, passes=(gate['joint_both'] >= 24 and
                    interval[0] > 0 and positive >= 12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('baseline', 'patch', 'select', 'confirm'), required=True)
    parser.add_argument('--split', choices=('discovery', 'confirmation'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-file', type=Path)
    parser.add_argument('--selection-file', type=Path)
    parser.add_argument('--inputs', type=Path, nargs='+')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    frozen = json.loads(CASES.read_text(encoding='utf-8'))
    if args.part in ('select', 'confirm'):
        if not args.baseline_file or not args.inputs:
            parser.error('Aggregation requires --inputs and --baseline-file')
        base = json.loads(args.baseline_file.read_text(encoding='utf-8'))
        chunks = [json.loads(path.read_text(encoding='utf-8')) for path in args.inputs]
        if args.part == 'select':
            if base['split'] != 'discovery' or any(c['split'] != 'discovery' for c in chunks):
                parser.error('Discovery-only inputs required for selection')
            result = select(chunks, base)
        else:
            if not args.selection_file:
                parser.error('Confirmation requires --selection-file')
            selected = json.loads(args.selection_file.read_text(encoding='utf-8'))
            if selected['selected'] not in LAYERS or base['split'] != 'confirmation':
                parser.error('No selected discovery layer or wrong baseline split')
            if any(c['split'] != 'confirmation' or c['layers'] != [selected['selected']] for c in chunks):
                parser.error('Confirmation layer or split differs from selected discovery layer')
            rows = [r for chunk in chunks for r in chunk['patches']]
            controls = [c for chunk in chunks for c in chunk['controls']]
            if len([c for c in controls if c['kind'] == 'self']) != 64:
                raise ValueError('Incomplete self-patch controls on confirmation')
            result = confirm(rows, base, selected['selected'])
            result['readout_control'] = next(c for c in selected['controls'] if c['kind'] == 'readout')
            result['max_self_error'] = max(c['error'] for c in controls if c['kind'] == 'self')
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if not args.split or args.threads < 1:
        parser.error('Inference requires --split and positive --threads')
    pairs = frozen[args.split]
    if args.part == 'patch' and not args.baseline_file:
        parser.error('Patch needs --baseline-file')
    torch.set_num_threads(args.threads)
    model = Model(frozen['model'], revision=frozen['revision'])
    if model.revision != frozen['revision']:
        raise ValueError('Model revision changed')
    if args.part == 'baseline':
        result = dict(split=args.split, model=frozen['model'], revision=model.revision,
                      pairs=pairs, baseline=baseline(model, pairs))
        result['quality'] = quality(result['baseline'], len(pairs))
    else:
        base = json.loads(args.baseline_file.read_text(encoding='utf-8'))
        if base['split'] != args.split or base['revision'] != model.revision or base['pairs'] != pairs:
            raise ValueError('Saved baseline, split or model revision differs from frozen experiment')
        if args.split == 'discovery' and base['quality']['joint_both'] < 12:
            raise ValueError('Discovery did not pass its baseline gate')
        layers = LAYERS if args.split == 'discovery' else None
        if args.split == 'confirmation':
            if not args.selection_file:
                parser.error('Confirmation patch requires --selection-file')
            selected = json.loads(args.selection_file.read_text(encoding='utf-8'))
            layers = [selected['selected']]
            if layers[0] not in LAYERS or selected['quality']['joint_both'] < 12:
                raise ValueError('No eligible discovery selection')
        start, stop = args.start, args.stop if args.stop is not None else len(pairs)
        if start < 0 or stop > len(pairs) or start >= stop:
            parser.error('Invalid pair range')
        rows, controls = patch_chunk(model, pairs, base['baseline'], start, stop, layers,
                                     args.split == 'discovery')
        result = dict(split=args.split, model=frozen['model'], revision=model.revision,
                      start=start, stop=stop, layers=list(layers), patches=rows, controls=controls)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'split':args.split, 'part':args.part,
                      'quality': result.get('quality'),
                      'rows':len(result.get('patches', [])), 'output':str(args.out)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
