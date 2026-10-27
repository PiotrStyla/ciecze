#!/usr/bin/env python3
"""Eksperyment zaworu pytania w GoLLeM v6, bez zmiany faktow w parze.

python valve_flow.py --baseline-only --out valve_out
python valve_flow.py --out valve_out
Szybki pomiar: --fillers 0 --layers 0 19 --out valve_smoke

Wymaga dye_flow.py i bernoulli_layers.py obok skryptu. Korzysta z ich
ladowania modelu, przechwytywania stanow i interwencji, bez treningu wag.

Kazda para ma IDENTYCZNY opis i prefiks odpowiedzi 'Odpowiedz:'. Zmienia
sie tylko nazwa przedmiotu w pytaniu: kluczyk/kubek (po 2 tokeny).
Kolory czerwony/niebieski sa zamieniane miedzy przedmiotami, a kolejnosc
faktow i liczba zdan pomiedzy nimi sa kontrolowane niezaleznie.

Model jest przyczynowy: pozniejsze pytanie NIE MOZE zmieniac wczesniejszych
stanow opisu. Dlatego testujemy selekcje przy odczycie, nie wsteczne
przekierowanie calego przeplywu. facts_control musi pozostac identyczny.

Slad: ||h_key-h_cup|| / srednia norm stanow. Nie jest iloscia informacji.
Wynik: log P(czerwony) - log P(niebieski), pelne odpowiedzi 2-tokenowe,
z teacher forcing. Nie jest to trafnosc swobodnego generowania.
Interwencje: oba kierunki, dawca i losowo permutowana roznica o tej samej
normie, na wyjsciach probkowanych blokow PRZED koncowa normalizacja.
Boczny value_residual v0 pozostaje z przebiegu biorcy.
Efekt 'toward_donor' oznacza ruch ku faktycznemu wynikowi dawcy, ktory
moze byc BLEDNY. Nie jest automatycznie miara poprawnego przekierowania.

Artefakty: baseline.json (rowniez przy --baseline-only), summary.json,
traces.csv, patches.csv, PNG kazdego przypadku i overview.png. Stary pilot
z pytaniami roznej dlugosci pozostaje w pilot.json; nie jest nadpisywany.
To eksploracja na jednym checkpointcie i rodzinie szablonow. Jedno losowanie
null na interwencje; brak p-value i twierdzen o unikalnej sciezce przyczynowej.
Silny efekt patchingu koncowego odczytu jest oczekiwany konstrukcyjnie.
Podczas pomiaru sprawdzane sa: zgodnosc faktow i pozycji pary, maska
przyczynowa, niezmiennosc po self-patchingu i zgodnosc norm null/dawcy.
Przy --baseline-only kontrole wymagajace interwencji maja wartosc null
w JSON: nie byly wykonane. Nie nalezy interpretowac tego jako bledu zero.

Zrodla: https://rome.baulab.info/ ; https://arxiv.org/abs/1906.02762
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bernoulli_layers import _local_dir, _import_gollem
from dye_flow import Experiment, write_csv


def make_pair(fillers, key_first, swapped):
    key_color, cup_color = ('niebieski', 'czerwony') if swapped else ('czerwony', 'niebieski')
    key_fact = f'Kluczyk jest {key_color}. '
    cup_fact = f'Kubek jest {cup_color}. '
    bridge = 'Na stole leży książka. ' * fillers
    context = key_fact + bridge + cup_fact if key_first else cup_fact + bridge + key_fact
    texts = [context + f'Jaki jest {obj}? Odpowiedź:' for obj in ['kluczyk', 'kubek']]
    return context, texts


def positions(tok, context, texts):
    encoded = [tok.encode(t) for t in texts]
    ids = [e.ids for e in encoded]
    if len(ids[0]) != len(ids[1]):
        raise ValueError('Pytania nie sa wyrownane tokenowo.')
    changed = [i for i, (a, b) in enumerate(zip(*ids)) if a != b]
    if not changed or any(encoded[0].offsets[i][0] < len(context) for i in changed):
        raise ValueError('Zmiana dotyczy opisu lub brakuje zmiany pytania.')
    if texts[0][:len(context)] != texts[1][:len(context)]:
        raise ValueError('Opisy nie sa identyczne.')
    if ids[0][max(changed) + 1:] != ids[1][max(changed) + 1:]:
        raise ValueError('Prefiks odpowiedzi nie jest identyczny.')
    facts = []
    for color in ['czerwony', 'niebieski']:
        start = context.index(color)
        stop = start + len(color)
        facts.extend(i for i, (a, b) in enumerate(encoded[0].offsets) if a < stop and b > start)
    mark = max(changed) + 1
    if mark >= len(ids[0]) - 1 or tok.decode([ids[0][mark]]) != '?':
        raise ValueError('Nie znaleziono wyrownanego znaku pytania przed odczytem.')
    regions = dict(question_subject=changed, question_mark=[mark],
                   readout=[len(ids[0]) - 1], facts_control=sorted(facts))
    return encoded, regions


def baseline(exp, case, count, key_first, swapped):
    context, texts = make_pair(count, key_first, swapped)
    encoded, regions = positions(exp.tok, context, texts)
    runs = [exp.run(text, capture=True) for text in texts]
    margins = [r[0] for r in runs]
    states = [r[1] for r in runs]
    signs = [-1, 1] if swapped else [1, -1]
    b = dict(case=case, fillers=count, key_first=key_first, swapped=swapped,
             context=context, prompts=texts, prompt_ids=[e.ids for e in encoded],
             expected_answers=['niebieski', 'czerwony'] if swapped else ['czerwony', 'niebieski'],
             margins_red_minus_blue=margins,
             correct_preferences=[m * sign > 0 for m, sign in zip(margins, signs)],
             signed_question_gap=signs[0] * (margins[0] - margins[1]), regions=regions,
             max_earlier_relative_difference=0.0,
             max_self_patch_error=None, max_null_norm_relative_error=None)
    trace = []
    for state, (a, other) in enumerate(zip(*states)):
        q = ((a - other).norm(dim=-1) /
             ((a.norm(dim=-1) + other.norm(dim=-1)) / 2).clamp_min(1e-8)).cpu().numpy()
        error = float(q[:min(regions['question_subject'])].max())
        b['max_earlier_relative_difference'] = max(b['max_earlier_relative_difference'], error)
        if error > 1e-5:
            raise AssertionError('Pozniejsze pytanie zmienia wczesniejsze pozycje.')
        for pos, sensitivity in enumerate(q):
            trace.append(dict(case=case, state=state, token=pos,
                              key_token=exp.tok.decode([encoded[0].ids[pos]]),
                              cup_token=exp.tok.decode([encoded[1].ids[pos]]),
                              sensitivity=float(sensitivity)))
    return b, states, signs, trace


def intervene(exp, b, states, signs, layers, seed):
    rows = []
    b['max_self_patch_error'] = 0.0
    b['max_null_norm_relative_error'] = 0.0
    gen = torch.Generator(device=exp.device).manual_seed(seed)
    margins = b['margins_red_minus_blue']
    for donor, recipient in [(0, 1), (1, 0)]:
        gap = margins[donor] - margins[recipient]
        for layer in layers:
            for region, pos in b['regions'].items():
                source, target = states[donor][layer + 1][pos], states[recipient][layer + 1][pos]
                delta = source - target
                permutation = torch.randperm(delta.shape[-1], generator=gen, device=exp.device)
                null = target + delta[:, permutation]
                expected = delta.norm(dim=-1)
                norm_error = float(((expected - (null - target).norm(dim=-1)).abs() /
                                    expected.clamp_min(1e-8)).max())
                b['max_null_norm_relative_error'] = max(b['max_null_norm_relative_error'], norm_error)
                if norm_error > 1e-4:
                    raise AssertionError('Null nie zachowuje normy perturbacji.')
                for kind, value in [('donor', source), ('norm_matched_null', null)]:
                    margin, _ = exp.run(b['prompts'][recipient], patch=(layer, pos, value))
                    shift = margin - margins[recipient]
                    if region == 'facts_control' and abs(shift) > 1e-4:
                        raise AssertionError('Identyczne stany opisu zmieniaja wynik.')
                    rows.append(dict(case=b['case'], direction='key_to_cup' if donor == 0 else 'cup_to_key',
                        layer=layer, region=region, kind=kind,
                        recipient_margin=margins[recipient], donor_margin=margins[donor],
                        patched_margin=margin, toward_donor=shift * float(np.sign(gap)),
                        gap_fraction=shift / gap if abs(gap) >= 0.1 else None,
                        prefers_donor_expected_answer=margin * signs[donor] > 0,
                        perturbation_norm=float((value - target).norm())))
            pos = b['regions']['readout']
            score, _ = exp.run(b['prompts'][recipient], patch=(layer, pos, states[recipient][layer + 1][pos]))
            b['max_self_patch_error'] = max(b['max_self_patch_error'], abs(score - margins[recipient]))
            if abs(score - margins[recipient]) > 1e-4:
                raise AssertionError('Self-patching zmienia wynik.')
    return rows


def plot_results(out, baselines, traces, patches, layers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(baselines))
    ax.bar(x - .18, [b['margins_red_minus_blue'][0] for b in baselines], .36, label='question: key')
    ax.bar(x + .18, [b['margins_red_minus_blue'][1] for b in baselines], .36, label='question: cup')
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set_xticks(x, [b['case'] for b in baselines], rotation=45, ha='right')
    ax.set(ylabel='log P(red) - log P(blue)', title='Same facts, different question; r/b = key color')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'overview.png', dpi=130); plt.close(fig)
    for b in baselines:
        case = b['case']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        selected = [r for r in traces if r['case'] == case]
        grid = np.zeros((max(r['state'] for r in selected) + 1, len(b['prompt_ids'][0])))
        for r in selected:
            grid[r['state'], r['token']] = r['sensitivity']
        image = axes[0].imshow(grid, aspect='auto', origin='lower', interpolation='nearest')
        axes[0].set(title='Relative sensitivity to question', xlabel='token position', ylabel='state, before final norm')
        fig.colorbar(image, ax=axes[0])
        for pos in b['regions']['question_subject']:
            axes[0].axvline(pos, color='white', linestyle=':', linewidth=.8)
        regions = list(b['regions'])
        grids = []
        for direction in ['key_to_cup', 'cup_to_key']:
            effect = np.zeros((len(layers), len(regions)))
            for i, layer in enumerate(layers):
                for j, region in enumerate(regions):
                    rows = [r for r in patches if r['case'] == case and r['direction'] == direction
                            and r['layer'] == layer and r['region'] == region]
                    actual = next(r['toward_donor'] for r in rows if r['kind'] == 'donor')
                    null = next(r['toward_donor'] for r in rows if r['kind'] == 'norm_matched_null')
                    effect[i, j] = actual - null
            grids.append(effect)
        limit = max(float(np.abs(g).max()) for g in grids) or 1e-8
        for ax, grid, direction in zip(axes[1:], grids, ['key -> cup', 'cup -> key']):
            image = ax.imshow(grid, aspect='auto', origin='lower', cmap='coolwarm', vmin=-limit, vmax=limit)
            ax.set(title=direction + ': donor minus null', ylabel='sampled block (zero-based)')
            ax.set_xticks(range(len(regions)), regions, rotation=35, ha='right')
            ax.set_yticks(range(len(layers)), layers)
            fig.colorbar(image, ax=ax, label='margin shift toward donor')
        fig.suptitle(case + ' | identical context; exploratory causal interventions')
        fig.tight_layout(); fig.savefig(out / f'{case}.png', dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='PiotrSty/gollem-v6-250m-sft-answers')
    p.add_argument('--out', type=Path, default=Path('valve_out'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--fillers', type=int, nargs='+', default=[0, 2, 6])
    p.add_argument('--layers', type=int, nargs='+')
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--baseline-only', action='store_true')
    args = p.parse_args()
    if args.threads < 1 or any(n < 0 for n in args.fillers) or len(set(args.fillers)) != len(args.fillers):
        p.error('Threads must be positive and fillers distinct nonnegative integers.')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    path = _local_dir(args.model, ['config.json', 'model.safetensors', 'tokenizer.json', 'modeling_gollem_v6.py'])
    model, tok = _import_gollem(path).load_gollem_v6(path, args.device)
    exp = Experiment(model, tok, args.device)
    n = len(model.blocks)
    layers = args.layers if args.layers is not None else sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    if len(set(layers)) != len(layers) or any(i < 0 or i >= n for i in layers):
        p.error(f'Layers must be distinct indices in 0..{n-1}.')
    args.out.mkdir(parents=True, exist_ok=True)
    baselines, traces, patches = [], [], []
    for count in args.fillers:
        for key_first in [True, False]:
            for swapped in [False, True]:
                case = f"{'key_first' if key_first else 'cup_first'}_f{count}_{'b' if swapped else 'r'}"
                b, states, signs, trace = baseline(exp, case, count, key_first, swapped)
                baselines.append(b); traces.extend(trace)
                if not args.baseline_only:
                    patches.extend(intervene(exp, b, states, signs, layers, args.seed + len(baselines)))
                print(f"{case}: margins={b['margins_red_minus_blue']} correct={b['correct_preferences']}", flush=True)
    summary = dict(model=args.model, snapshot=str(path), torch=torch.__version__, device=args.device,
                   seed=args.seed, sampled_blocks=layers, baselines=baselines,
                   limitations=['Exploratory single-model experiment; template chosen after initial pilot.',
                     'Kluczyk replaces klucz to align query token counts, not to optimize accuracy.',
                     'One norm-matched null per intervention; no significance claims.',
                     'Identical earlier context states follow causality, not a discovered mechanism.',
                     'Late readout patches are an expected calibration, not proof of a unique route.',
                     'Only residual stream is patched; recipient value_residual v0 is unchanged.',
                     'Forced two-answer preference is not free-generation accuracy.'])
    paired = sum(all(b['correct_preferences']) for b in baselines)
    summary['paired_preference_passes'] = paired
    (args.out / 'baseline.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.baseline_only:
        (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        write_csv(args.out / 'traces.csv', traces); write_csv(args.out / 'patches.csv', patches)
        plot_results(args.out, baselines, traces, patches, layers)
    print(f'Paired preferences: {paired}/{len(baselines)}. Output: {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
