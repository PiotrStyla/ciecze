#!/usr/bin/env python3
"""Independent color-binding task gate and paired same-fact/opposite-fact patching.

See docs/HELDOUT_PROTOCOL.md. This script never trains or alters model weights.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from bernoulli_layers import _import_gollem

GOLLEM = 'PiotrSty/gollem-v6-250m-sft-answers'
REVISION = '3a60edc280d93fb73ead0f5a5ba220986525aa76'
CANDIDATES = ('czerwony', 'niebieski')
LAYERS = (5, 10)


def words(pair, swapped):
    return (CANDIDATES[swapped], CANDIDATES[1 - swapped])


def description(pair, swapped, paraphrase=False):
    a, b = pair
    ca, cb = words(pair, swapped)
    if paraphrase:
        return f'{a.capitalize()} ma kolor {ca}. {b.capitalize()} ma kolor {cb}. '
    return f'{a.capitalize()} jest {ca}. {b.capitalize()} jest {cb}. '


def question(pair, target, swapped, paraphrase=False):
    return description(pair, swapped, paraphrase) + (
        f'Jaki kolor ma {pair[target]}? Odpowiedz jednym słowem:')


class Model:
    def __init__(self, name, device='cpu', revision=None):
        self.name, self.device = name, device
        revision = revision or (REVISION if name == GOLLEM else None)
        files = (['config.json', 'tokenizer.json', 'model.safetensors', 'modeling_gollem_v6.py']
                 if name == GOLLEM else ['config.json', 'generation_config.json', 'tokenizer*', 'model.safetensors',
                                         'model.safetensors.index.json', 'model-*.safetensors'])
        self.snapshot = (str(Path(name).resolve()) if Path(name).is_dir() else
                         snapshot_download(name, revision=revision, allow_patterns=files))
        self.gollem = name == GOLLEM
        if self.gollem:
            module = _import_gollem(self.snapshot)
            self.model, self.tok = module.load_gollem_v6(self.snapshot, device)
            self.blocks = self.model.blocks
            self.eos = None
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tok = AutoTokenizer.from_pretrained(self.snapshot, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.snapshot, dtype=torch.float32).to(device).eval()
            self.blocks = self.model.model.layers
            self.eos = self.tok.eos_token_id
        self.revision = Path(self.snapshot).name if not Path(name).is_dir() else 'local-directory'

    def prefix(self, text):
        if self.gollem:
            return text
        return self.tok.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)

    def encode(self, text):
        if self.gollem:
            e = self.tok.encode(text)
            return e.ids, e.offsets
        e = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return e.input_ids, e.offset_mapping

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True) if not self.gollem else self.tok.decode(ids)

    def logits(self, ids):
        out = self.model(torch.tensor([ids], device=self.device))
        return out if self.gollem else out.logits

    def fact_end_spans(self, prompt):
        prefix = self.prefix(prompt)
        ids, offsets = self.encode(prefix)
        if not self.gollem:
            direct = self.tok.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
            if ids != direct:
                raise ValueError('Chat prefix tokenization differs from template tokenization')
        beginning = prefix.find(prompt)
        if beginning < 0:
            raise ValueError('Question text absent from formatted prefix')
        ends = [i for i, char in enumerate(prompt) if char == '.']
        if len(ends) != 2:
            raise ValueError('Expected two fact-final periods')
        positions = []
        for end in ends:
            absolute = beginning + end
            hits = [i for i, (left, right) in enumerate(offsets)
                    if left <= absolute < right]
            if len(hits) != 1:
                raise ValueError('Fact-final punctuation must map to one token')
            positions.append(hits)
        if positions[0] == positions[1]:
            raise ValueError('Fact ends overlap')
        return positions

    @torch.inference_mode()
    def states(self, prompt):
        ids, _ = self.encode(self.prefix(prompt))
        states, handles = [], []
        for block in self.blocks:
            def collect(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                states.append(hidden[0].detach().clone())
            handles.append(block.register_forward_hook(collect))
        try:
            self.logits(ids)
        finally:
            for handle in handles:
                handle.remove()
        return states

    def patch_hook(self, layer, positions, replacements):
        # Only fact-color positions; other positions and side outputs stay with recipient.
        def inject(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            for dest, source in zip(positions, replacements):
                patched[:, dest, :] = source.to(patched.device)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched
        return self.blocks[layer].register_forward_hook(inject)

    @torch.inference_mode()
    def margin(self, prompt, patch=None):
        prefix = self.prefix(prompt)
        start, _ = self.encode(prefix)
        scores = []
        handle = self.patch_hook(*patch) if patch is not None else None
        try:
            for color in CANDIDATES:
                suffix = (' ' if self.gollem else '') + color
                full, _ = self.encode(prefix + suffix)
                if full[:len(start)] != start or len(full) == len(start):
                    raise ValueError('Answer changes prefix tokenization or is empty')
                logits = self.logits(full)
                start_idx = len(start)
                log_probs = logits[0, start_idx - 1:-1].float().log_softmax(-1)
                target = torch.tensor(full[start_idx:], device=self.device)
                scores.append(float(log_probs.gather(-1, target[:, None]).sum()))
        finally:
            if handle is not None:
                handle.remove()
        return scores[0] - scores[1]

    @torch.inference_mode()
    def free_answer(self, prompt, max_tokens=8):
        ids, _ = self.encode(self.prefix(prompt))
        generated = []
        for _ in range(max_tokens):
            token = int(self.logits(ids)[0, -1].argmax())
            if token == self.eos:
                break
            generated.append(token)
            ids.append(token)
        return self.decode(generated)


def free_color(text):
    cleaned = text.strip().strip('"\'„”').strip()
    match = re.fullmatch(r'(czerwony|niebieski)[.!?,;:]?', cleaned, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def baseline(model, pairs):
    rows = []
    for idx, pair in enumerate(pairs):
        for swapped in (0, 1):
            for target in (0, 1):
                prompt = question(pair, target, swapped)
                margin = model.margin(prompt)
                free = model.free_answer(prompt)
                expected = words(pair, swapped)[target]
                forced = CANDIDATES[0] if margin > 0 else CANDIDATES[1]
                rows.append(dict(pair=idx, objects=pair, swapped=swapped, target=target,
                                 prompt=prompt, expected=expected, margin=margin,
                                 forced=forced, free=free, free_parsed=free_color(free),
                                 forced_ok=forced == expected, free_ok=free_color(free) == expected))
        print(f'baseline {idx + 1}/{len(pairs)}: {pair}', flush=True)
    return rows


def score_gate(rows):
    contexts = {}
    for row in rows:
        contexts.setdefault((row['pair'], row['swapped']), []).append(row)
    forced = sum(len(rr) == 2 and all(r['forced_ok'] for r in rr) for rr in contexts.values())
    free = sum(len(rr) == 2 and all(r['free_ok'] for r in rr) for rr in contexts.values())
    return dict(contexts=len(contexts), forced_both=forced, free_both=free,
                passes=len(contexts) == 24 and forced >= 18 and free >= 18)


def patching(model, pairs):
    rows = []
    for idx, pair in enumerate(pairs):
        for swapped in (0, 1):
            for target in (0, 1):
                recipient = question(pair, target, swapped)
                same = question(pair, target, swapped, paraphrase=True)
                opposite = question(pair, target, 1 - swapped, paraphrase=True)
                spans = {kind: model.fact_end_spans(text) for kind, text in
                         [('recipient', recipient), ('same', same), ('opposite', opposite)]}
                states = {kind: model.states(text) for kind, text in
                          [('recipient', recipient), ('same', same), ('opposite', opposite)]}
                base = model.margin(recipient)
                direction = 1 if words(pair, swapped)[target] == CANDIDATES[0] else -1
                for layer in LAYERS:
                    replacements = {}
                    norms = {}
                    for kind in ('recipient', 'same', 'opposite'):
                        replacements[kind] = [states[kind][layer][span] for span in spans[kind]]
                        if kind != 'recipient':
                            norms[kind] = float(torch.cat([
                                (source - reference).reshape(-1)
                                for source, reference in zip(replacements[kind], replacements['recipient'])
                            ]).norm())
                    positions = spans['recipient']
                    self_margin = model.margin(recipient, patch=(layer, positions, replacements['recipient']))
                    if abs(self_margin - base) > 1e-4:
                        raise AssertionError(f'Self-patch changed score: {self_margin - base}')
                    same_margin = model.margin(recipient, patch=(layer, positions, replacements['same']))
                    opposite_margin = model.margin(recipient, patch=(layer, positions, replacements['opposite']))
                    rows.append(dict(pair=idx, objects=pair, swapped=swapped, target=target, layer=layer,
                                     recipient=recipient, same=same, opposite=opposite,
                                     recipient_positions=positions, same_positions=spans['same'],
                                     opposite_positions=spans['opposite'],
                                     base_margin=base, self_margin=self_margin,
                                     same_margin=same_margin, opposite_margin=opposite_margin,
                                     same_norm=norms['same'], opposite_norm=norms['opposite'],
                                     correct_color=words(pair, swapped)[target],
                                     effect=direction * (same_margin - opposite_margin)))
        print(f'patching {idx + 1}/{len(pairs)}: {pair}', flush=True)
    return rows


def analyze(rows, seed=20261028):
    by_pair = []
    for idx in sorted({r['pair'] for r in rows}):
        values = [r['effect'] for r in rows if r['pair'] == idx and r['layer'] == 5]
        if len(values) != 4:
            raise ValueError('Incomplete primary-effect grid')
        by_pair.append(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    data = np.array(by_pair)
    sample = rng.choice(data, size=(10000, len(data)), replace=True).mean(axis=1)
    return dict(pair_effects=by_pair, mean_effect=float(data.mean()),
                bootstrap_95=[float(v) for v in np.percentile(sample, [2.5, 97.5])],
                positive_pairs=int((data > 0).sum()), pair_count=len(data))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=GOLLEM)
    p.add_argument('--split', choices=('gate', 'heldout'), required=True)
    p.add_argument('--patch', action='store_true', help='Only heldout; requires independent gate file')
    p.add_argument('--gate-result', type=Path, help='Passing calibration JSON for --patch')
    p.add_argument('--revision', help='Pinned Hub commit; defaults to GoLLeM research revision')
    p.add_argument('--baseline-result', type=Path, help='Recorded heldout baseline for --patch')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1 or args.patch and (args.split != 'heldout' or not args.gate_result or
                                            not args.baseline_result):
        p.error('threads must be positive; patch needs heldout, --gate-result and --baseline-result')
    torch.set_num_threads(args.threads)
    data = json.loads((Path(__file__).parent / 'heldout_cases.json').read_text(encoding='utf-8'))
    pairs = data[args.split]
    gate = None
    if args.patch:
        gate = json.loads(args.gate_result.read_text(encoding='utf-8'))
        if gate['model'] != args.model or not gate['gate']['passes']:
            p.error('Selected model must have passed its own gate')
    revision = args.revision or (gate['revision'] if gate else None)
    model = Model(args.model, revision=revision)
    if gate and gate['revision'] != model.revision:
        p.error('Gate checkpoint revision differs from intervention checkpoint')
    prior = None
    if args.baseline_result:
        prior = json.loads(args.baseline_result.read_text(encoding='utf-8'))
        if prior['model'] != args.model or prior['revision'] != model.revision or (
                prior['split'] != 'heldout' or prior['pairs'] != pairs):
            p.error('Baseline checkpoint or frozen heldout pairs differ')
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(model=args.model, revision=model.revision, snapshot=model.snapshot,
                  torch=torch.__version__, seed=data['seed'], split=args.split,
                  pairs=pairs, baseline=prior['baseline'] if prior else baseline(model, pairs))
    if args.split == 'gate':
        result['gate'] = score_gate(result['baseline'])
    else:
        by_context = {}
        for row in result['baseline']:
            by_context.setdefault((row['pair'], row['swapped']), []).append(row)
        result['heldout_quality'] = dict(contexts=len(by_context),
            forced_both=sum(all(r['forced_ok'] for r in rr) for rr in by_context.values()),
            free_both=sum(all(r['free_ok'] for r in rr) for rr in by_context.values()))
        if args.patch:
            result['patches'] = patching(model, pairs)
            result['analysis'] = analyze(result['patches'], data['seed'])
            result['analysis']['meets_rule'] = (result['analysis']['bootstrap_95'][0] > 0 and
                result['analysis']['positive_pairs'] >= 15 and
                result['heldout_quality']['forced_both'] >= 30 and
                result['heldout_quality']['free_both'] >= 30)
    (args.out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(model=args.model, split=args.split, gate=result.get('gate'),
                          quality=result.get('heldout_quality'), analysis=result.get('analysis')),
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
