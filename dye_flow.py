#!/usr/bin/env python3
"""Dwa barwniki w GoLLeM: slad zmiany faktu i dystraktora + causal patching.

Wymaga bernoulli_layers.py w tym samym katalogu i jego zaleznosci.
Uruchomienie: python dye_flow.py --out dye_out
Szybka proba: python dye_flow.py --fillers 0 --layers 0 19 --out dye_smoke

Swiadomie tylko natywny GoLLeM v6. Ladowanie korzysta z istniejacego kodu
bernoulli_layers.py; nie zmienia ani nie trenuje wag. Kod checkpointa jest
wykonywany tak jak w poprzednim eksperymencie.

Stan 0 = wejscie pierwszego bloku; stan l+1 = wyjscie bloku l PRZED lnf.
Patching dotyczy calego residual stream wskazanych pozycji, nie sciezek
uwagi ani bocznego value_residual v0 (pozostaje z przebiegu biorcy).
Slad to wzgledna wrazliwosc, NIE zachowana masa ani ilosc informacji.

Odpowiedzi sa dwutokenowe: wynik to roznica sum log-prawdopodobienstw
pelnych odpowiedzi czerwony/niebieski, z teacher forcing. Pary wymagaja
identycznych pozycji tokenow. Null: przetasowanie wspolrzednych roznicy
aktywacji, zachowujace norme perturbacji na kazdej pozycji. Jedna realizacja
null na interwencje; nie sluzy do wyznaczania p-value.

Szablon wybrano po pilocie na tym samym checkpointcie. To eksploracja,
nie niezalezny benchmark. Warstwy patchingu sa domyslnie probkowane (5).

Artefakty: summary.json (prompty, tokeny, snapshot, marginesy i kontrole),
traces.csv (wszystkie pozycje i stany), patches.csv (oba kierunki i null),
PNG dla kazdego wariantu. pilot.json z pierwszego badania zapisano obok
wynikow; kolejne uruchomienia nie odtwarzaja ani nie nadpisuja pilota.
Kontrole wykonywane podczas pomiaru: brak wplywu na wczesniejsze tokeny,
niezmiennosc po self-patchingu i zgodnosc norm rzeczywistych perturbacji.
Duzy efekt patchingu ostatniego stanu odczytu jest oczekiwany; sam nie
stanowi odkrycia kanalu transportu. Kolory mapy przyczynowej pokazuja
efekt dawcy minus null, usredniony po obu kierunkach, nie p-value.

Zrodla: https://arxiv.org/abs/1906.02762
https://aclanthology.org/2020.acl-main.385/ ; https://rome.baulab.info/
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from bernoulli_layers import _local_dir, _import_gollem


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prompts(fillers, order):
    def make(signal, distractor):
        fact = f'Klucz jest {signal}. '
        noise = f'Kubek jest {distractor}. '
        bridge = 'Na stole leży książka. ' * fillers
        text = fact + bridge + noise if order == 'source_first' else noise + bridge + fact
        return text + 'Jaki jest klucz? Klucz jest'
    return [make('czerwony', 'zielony'), make('niebieski', 'zielony'),
            make('czerwony', 'niebieski')]


class Experiment:
    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device
        self.answers = [' czerwony', ' niebieski']

    def inputs(self, prompt):
        prefix = self.tok.encode(prompt).ids
        full = [self.tok.encode(prompt + a).ids for a in self.answers]
        if any(ids[:len(prefix)] != prefix for ids in full):
            raise ValueError('Tokenizacja odpowiedzi zmienia koniec promptu.')
        if len(full[0]) != len(full[1]) or len(full[0]) == len(prefix):
            raise ValueError('Eksperyment wymaga rownej, niezerowej dlugosci odpowiedzi.')
        if len(full[0]) > self.model.block:
            raise ValueError('Prompt i odpowiedz przekraczaja okno modelu.')
        return torch.tensor(full, device=self.device), len(prefix)

    @torch.inference_mode()
    def run(self, prompt, capture=False, patch=None):
        ids, T = self.inputs(prompt)
        states, handles = [], []
        if capture:
            def first(module, args):
                states.append(args[0][0, :T].detach().clone())
            handles.append(self.model.blocks[0].register_forward_pre_hook(first))
            def collect(module, args, output):
                # Future answer tokens must never alter the captured prefix.
                x = output[0]
                error = (x[0, :T] - x[1, :T]).norm(dim=-1)
                scale = x[0, :T].norm(dim=-1).clamp_min(1e-8)
                if float((error / scale).max()) > 1e-5:
                    raise AssertionError('Answer tokens alter the captured prefix.')
                states.append(x[0, :T].detach().clone())
            handles.extend(b.register_forward_hook(collect) for b in self.model.blocks)
        if patch is not None:
            layer, positions, replacement = patch
            def inject(module, args, output):
                x, v0 = output
                x = x.clone()
                x[:, positions, :] = replacement.to(x.device)
                return x, v0
            handles.append(self.model.blocks[layer].register_forward_hook(inject))
        try:
            logits = self.model(ids)
            prediction = logits[:, T - 1:-1].float().log_softmax(-1)
            scores = prediction.gather(-1, ids[:, T:, None]).squeeze(-1).sum(-1)
            margin = float(scores[0] - scores[1])
            return margin, states
        finally:
            for handle in handles:
                handle.remove()


def aligned_positions(tok, texts):
    ids = [tok.encode(t).ids for t in texts]
    if len({len(row) for row in ids}) != 1:
        raise ValueError('Pary promptow nie sa wyrownane tokenowo.')
    signal = [i for i, (a, b) in enumerate(zip(ids[0], ids[1])) if a != b]
    noise = [i for i, (a, b) in enumerate(zip(ids[0], ids[2])) if a != b]
    if not signal or not noise or set(signal) & set(noise):
        raise ValueError('Niejednoznaczne pozycje barwnikow.')
    end = len(ids[0]) - 1
    # A matched single-position reference outside the changed spans and readout.
    candidates = [i for i in range(1, end) if i not in signal + noise]
    control = min(candidates, key=lambda i: abs(i - end // 2))
    return ids[0], {'source': signal, 'distractor': noise,
                    'midpoint_reference': [control], 'readout': [end]}


def run_case(exp, case_id, texts, layers, seed):
    ids, regions = aligned_positions(exp.tok, texts)
    runs = [exp.run(t, capture=True) for t in texts]
    margins = [r[0] for r in runs]
    states = [r[1] for r in runs]
    baseline = dict(case=case_id, prompts=texts, prompt_ids=[exp.tok.encode(t).ids for t in texts],
                    margins_red_minus_blue=margins,
                    red_preferred=margins[0] > 0, blue_preferred=margins[1] < 0,
                    signal_gap=margins[0] - margins[1],
                    distractor_shift=margins[2] - margins[0], regions=regions,
                    max_norm_control_relative_error=0.0)
    traces, patches = [], []
    generator = torch.Generator(device=exp.device).manual_seed(seed)
    for dye, other, changed in [('signal', 1, regions['source']),
                                 ('distractor', 2, regions['distractor'])]:
        for state, (a, b) in enumerate(zip(states[0], states[other])):
            delta = (a - b).norm(dim=-1)
            scale = ((a.norm(dim=-1) + b.norm(dim=-1)) / 2).clamp_min(1e-8)
            q = (delta / scale).cpu().numpy()
            # Causal masking: positions before the changed span cannot react.
            if changed[0] > 0 and float(q[:changed[0]].max()) > 1e-5:
                raise AssertionError('Zmiana wplywa na wczesniejsze pozycje.')
            for pos, sensitivity in enumerate(q):
                traces.append(dict(case=case_id, dye=dye, state=state, token=pos,
                                   token_text=exp.tok.decode([ids[pos]]),
                                   sensitivity=float(sensitivity)))
        for donor, recipient in [(0, other), (other, 0)]:
            gap = margins[donor] - margins[recipient]
            for layer in layers:
                for region, positions in regions.items():
                    target = states[recipient][layer + 1][positions]
                    source = states[donor][layer + 1][positions]
                    difference = source - target
                    permutation = torch.randperm(source.shape[-1], generator=generator, device=exp.device)
                    replacements = [('donor', source),
                                    ('norm_matched_null', target + difference[:, permutation])]
                    expected_norm = difference.norm(dim=-1)
                    null_norm = (replacements[1][1] - target).norm(dim=-1)
                    norm_error = float(((expected_norm - null_norm).abs()
                                        / expected_norm.clamp_min(1e-8)).max())
                    baseline['max_norm_control_relative_error'] = max(
                        baseline['max_norm_control_relative_error'], norm_error)
                    if norm_error > 1e-4:
                        raise AssertionError('Null perturbation does not preserve norm.')
                    for kind, replacement in replacements:
                        score, _ = exp.run(texts[recipient], patch=(layer, positions, replacement))
                        shift = score - margins[recipient]
                        patches.append(dict(case=case_id, dye=dye,
                            direction=f'{donor}_to_{recipient}', layer=layer, region=region, kind=kind,
                            recipient_margin=margins[recipient], donor_margin=margins[donor],
                            patched_margin=score, margin_shift=shift,
                            toward_donor=shift * float(np.sign(gap)),
                            gap_fraction=shift / gap if abs(gap) >= 0.1 else None,
                            perturbation_norm=float((replacement - target).norm())))
    # Real self-interventions must leave the output unchanged, including tuple v0.
    self_errors = []
    for layer in layers:
        positions = regions['readout']
        same, _ = exp.run(texts[0], patch=(layer, positions, states[0][layer + 1][positions]))
        self_errors.append(abs(same - margins[0]))
    baseline['max_self_patch_error'] = max(self_errors)
    if max(self_errors) > 1e-4:
        raise AssertionError('Self-patching zmienia predykcje.')
    return baseline, traces, patches


def plots(out, baselines, traces, patches, layers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for baseline in baselines:
        case = baseline['case']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        regions = list(baseline['regions'])
        for row, dye in enumerate(['signal', 'distractor']):
            selected = [r for r in traces if r['case'] == case and r['dye'] == dye]
            grid = np.zeros((1 + max(r['state'] for r in selected), len(baseline['prompt_ids'][0])))
            for r in selected:
                grid[r['state'], r['token']] = r['sensitivity']
            image = axes[row, 0].imshow(grid, aspect='auto', origin='lower', interpolation='nearest')
            axes[row, 0].set(title=f'{dye}: relative activation difference', xlabel='token position', ylabel='state (before final norm)')
            for name in ['source', 'distractor']:
                for pos in baseline['regions'][name]:
                    axes[row, 0].axvline(pos, color='white', linestyle=':', linewidth=0.8)
            fig.colorbar(image, ax=axes[row, 0])
            selected = [r for r in patches if r['case'] == case and r['dye'] == dye]
            effect = np.zeros((len(layers), len(regions)))
            for i, layer in enumerate(layers):
                for j, region in enumerate(regions):
                    rows = [r for r in selected if r['layer'] == layer and r['region'] == region]
                    genuine = np.mean([r['toward_donor'] for r in rows if r['kind'] == 'donor'])
                    null = np.mean([r['toward_donor'] for r in rows if r['kind'] != 'donor'])
                    effect[i, j] = genuine - null
            vmax = max(float(np.abs(effect).max()), 1e-8)
            image = axes[row, 1].imshow(effect, aspect='auto', origin='lower', cmap='coolwarm', vmin=-vmax, vmax=vmax)
            axes[row, 1].set(title=f'{dye}: donor minus null (mean two directions)', ylabel='patched block (zero-based)')
            axes[row, 1].set_xticks(range(len(regions)), regions, rotation=20, ha='right')
            axes[row, 1].set_yticks(range(len(layers)), layers)
            fig.colorbar(image, ax=axes[row, 1], label='log-likelihood margin shift toward donor')
        fig.suptitle(case + ' | exploratory sensitivity and causal interventions')
        fig.tight_layout()
        fig.savefig(out / f'{case}.png', dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='PiotrSty/gollem-v6-250m-sft-answers')
    parser.add_argument('--out', type=Path, default=Path('dye_out'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fillers', type=int, nargs='+', default=[0, 2, 6])
    parser.add_argument('--layers', type=int, nargs='+', help='zero-based block indices; default five sampled blocks')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    if args.threads < 1 or any(n < 0 for n in args.fillers) or len(set(args.fillers)) != len(args.fillers):
        parser.error('Threads must be positive; filler counts must be distinct and nonnegative.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    path = _local_dir(args.model, ['config.json','model.safetensors','tokenizer.json','modeling_gollem_v6.py'])
    model, tok = _import_gollem(path).load_gollem_v6(path, args.device)
    exp = Experiment(model, tok, args.device)
    n = len(model.blocks)
    layers = args.layers if args.layers is not None else sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    if len(set(layers)) != len(layers) or any(i < 0 or i >= n for i in layers):
        parser.error(f'Layers must be distinct indices in 0..{n-1}.')
    args.out.mkdir(parents=True, exist_ok=True)
    baselines, traces, patches = [], [], []
    for count in args.fillers:
        for order in ['source_first', 'source_last']:
            case = f'{order}_f{count}'
            baseline, t, p = run_case(exp, case, prompts(count, order), layers, args.seed + len(baselines))
            baselines.append(baseline)
            traces.extend(t)
            patches.extend(p)
            print(f"{case}: margins={baseline['margins_red_minus_blue']} "
                  f"paired_baseline_ok={baseline['red_preferred'] and baseline['blue_preferred']} "
                  f"self_error={baseline['max_self_patch_error']:.2g}", flush=True)
    write_csv(args.out / 'traces.csv', traces)
    write_csv(args.out / 'patches.csv', patches)
    summary = dict(model=args.model, snapshot=str(path), torch=torch.__version__,
                   device=args.device, seed=args.seed, sampled_blocks=layers,
                   sources=['https://arxiv.org/abs/1906.02762', 'https://rome.baulab.info/',
                            'https://aclanthology.org/2020.acl-main.385/'],
                   limitations=['Exploratory; template selected in a pilot on this checkpoint.',
                     'One norm-matched null per intervention; no significance claims.',
                     'Difference norm is sensitivity, not conserved mass or information.',
                     'Patches change residual stream only; recipient value_residual v0 is unchanged.',
                     'Positive patches do not establish unique or necessary causal paths.',
                     'Contrast is preference between two forced answers, not free-generation accuracy.'],
                   baselines=baselines)
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    plots(args.out, baselines, traces, patches, layers)
    passed = sum(b['red_preferred'] and b['blue_preferred'] for b in baselines)
    print(f'Paired baseline: {passed}/{len(baselines)}. Output: {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
