#!/usr/bin/env python3
"""Replikacja wiazania przedmiot-kolor na innych szablonach i parach slow.

python replicate_flow.py --part replicate --out runs/replicate
python replicate_flow.py --part nulls --null-draws 10 --out runs/nulls
Smoke: python replicate_flow.py --part replicate --pairs kluczyk/kubek --templates T1 --fillers 0

Wymaga dye_flow.py i bernoulli_layers.py. Nie zmienia wag. Domyslnie przypina
rewizje checkpointu z poprzednich etapow.

CZESC replicate (szybka, bez interwencji):
 4 szablony x 4 pary rzeczownikow (meskie, rowno tokenizowane) x 2 kolejnosc
 x 2 dlugosci wstawki = 64 uklady po 6 promptow. Dla kazdego liczymy
 R_key, R_cup, I i C jak w binding_flow.py. Kryteria rownowagi tokenow sa
 mechaniczne i sprawdzane przed pomiarem; nie odrzucamy ukladow po wynikach.
 Szablon/para, ktore nie przejda wyrownania, sa raportowane jako odrzucone
 i nie wchodza do statystyk.

 CZESC nulls (kalibracja, drozsza):
 pierwotna konfiguracja (T1, kluczyk/kubek, wstawki 0/2/6, oba porzadki),
 region tokenow kolorow faktow, warstwy 0/5/10/15. Dla kazdej interwencji
 dawcy losujemy R niezaleznych null per pytanie i liczymy rozklad
 selektywnego przesuniecia. p_emp = (1+#{|null|>=|donor|})/(R+1), dwustronne.
 Przy R=10 najmniejsze mozliwe p wynosi 1/11~0,09; to slaba kalibracja i nie
 jest to test potwierdzajacy dla malych efektow. Kontrole null sa liczone
 osobno dla kazdego pytania, bo prompty sie roznia.

Wynik odpowiedzi pozostaje kontrastem dwoch wymuszonych pelnych odpowiedzi,
nie trafnoscia swobodnego generowania. Jedno pytanie i jeden model.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from bernoulli_layers import _import_gollem
from dye_flow import Experiment, write_csv

REVISION = '3a60edc280d93fb73ead0f5a5ba220986525aa76'
NEUTRAL = 'opisany'
COLORS = ('czerwony', 'niebieski')

PAIRS = {'kluczyk/kubek': ('kluczyk', 'kubek'), 'kot/pies': ('kot', 'pies'),
         'rower/samochód': ('rower', 'samochód'), 'ołówek/zeszyt': ('ołówek', 'zeszyt')}

TEMPLATES = {
    'T1': {'fact': '{o} jest {c}', 'question': 'Jaki jest {o}? Odpowiedź:'},
    'T2': {'fact': '{o} jest {c}', 'question': 'Jaki kolor ma {o}? Odpowiedź:'},
    'T3': {'fact': '{o} jest {c}', 'question': 'Pytanie: {o}? Odpowiedź:'},
    'T4': {'fact': '{c} jest {o}', 'question': 'Jaki jest {o}? Odpowiedź:'},
}


def sentence(text):
    return text[0].upper() + text[1:] + '. '


def prompts(template, pair, fillers, key_first, condition):
    """Opis A/B/N (jedna zmiana przypisania kolorow) i oba pytania."""
    a, b = pair
    if condition == 'A':
        ca, cb = COLORS
    elif condition == 'B':
        ca, cb = COLORS[1], COLORS[0]
    else:
        ca = cb = NEUTRAL
    fact = template['fact']
    fact_a = sentence(fact.format(o=a, c=ca))
    fact_b = sentence(fact.format(o=b, c=cb))
    bridge = 'Na stole leży książka. ' * fillers
    context = fact_a + bridge + fact_b if key_first else fact_b + bridge + fact_a
    question = template['question']
    return context, [context + question.format(o=o) for o in (a, b)]


def regions(tok, ctx):
    """Regiony z warunku A; B i N musza roznic sie dokladnie na tokenach kolorow."""
    context, texts = ctx['A']
    encoded = [tok.encode(t) for t in texts]
    ids = [e.ids for e in encoded]
    if len(ids[0]) != len(ids[1]):
        return None, 'pytania nie sa wyrownane tokenowo'
    changed = [i for i, (x, y) in enumerate(zip(*ids)) if x != y]
    if not changed or any(encoded[0].offsets[i][0] < len(context) for i in changed):
        return None, 'zmiana wykracza poza nazwe przedmiotu w pytaniu'
    colors = []
    for word in COLORS:
        start = context.find(word)
        if start < 0:
            capitalized = word[0].upper() + word[1:]
            start = context.find(capitalized)
        if start < 0:
            return None, f'brak koloru {word} w opisie'
        colors.extend(i for i, (x, y) in enumerate(encoded[0].offsets)
                      if x < start + len(word) and y > start)
    if len(colors) != 4:
        return None, f'oczekiwano czterech tokenow kolorow, jest {len(colors)}'
    for condition in ['B', 'N']:
        other = [tok.encode(t).ids for t in ctx[condition][1]]
        for target in range(2):
            if len(ids[target]) != len(other[target]):
                return None, f'{condition}: inna dlugosc promptu niz A'
            if [i for i, (x, y) in enumerate(zip(ids[target], other[target])) if x != y] != sorted(colors):
                return None, f'{condition}: zmiana poza tokenami kolorow'
    return dict(fact_colors=sorted(colors), question_subject=changed,
                readout=[len(ids[0]) - 1]), None


def contexts(template, pair, fillers, key_first):
    out = {}
    for condition in ['A', 'B', 'N']:
        context, pair_texts = prompts(template, pair, fillers, key_first, condition)
        out[condition] = (context, pair_texts)
    return out


def contrasts(margins):
    key = margins['A'][0] - margins['B'][0]
    cup = margins['A'][1] - margins['B'][1]
    return dict(key_context_response=key, cup_context_response=cup,
                binding_interaction=(key - cup) / 2, common_context_effect=(key + cup) / 2,
                opposite_context_responses=key > 0 and cup < 0)


def run_replicate(exp, args):
    cases, rejected, traces = [], [], []
    for name, template in TEMPLATES.items():
        for pair_name, pair in PAIRS.items():
            for fillers in args.fillers:
                for key_first in [True, False]:
                    ctx = contexts(template, pair, fillers, key_first)
                    reg, reason = regions(exp.tok, ctx)
                    case = f'{name}|{pair_name}|f{fillers}|{"key" if key_first else "cup"}'
                    if reg is None:
                        rejected.append(dict(case=case, template=name, pair=pair_name,
                                             fillers=fillers, key_first=key_first, reason=reason))
                        continue
                    margins, states = {}, {}
                    for condition, (context, texts) in ctx.items():
                        runs = [exp.run(t, capture=True) for t in texts]
                        margins[condition] = [r[0] for r in runs]
                        states[condition] = [r[1] for r in runs]
                    b = dict(case=case, template=name, pair=pair_name, fillers=fillers,
                             key_first=key_first, prompts={c: v[1] for c, v in ctx.items()},
                             prompt_ids={c: [exp.tok.encode(t).ids for t in v[1]] for c, v in ctx.items()},
                             margins=margins, regions=reg,
                             **contrasts(margins))
                    cases.append(b)
                    for target, question in enumerate(['key', 'cup']):
                        for condition in ['B', 'N']:
                            for state, (x, other) in enumerate(zip(states['A'][target], states[condition][target])):
                                q = ((x - other).norm(dim=-1) /
                                     ((x.norm(dim=-1) + other.norm(dim=-1)) / 2).clamp_min(1e-8)).cpu().numpy()
                                if q[:min(b['regions']['fact_colors'])].max() > 1e-5:
                                    raise AssertionError('Zmiana opisu zmienia wczesniejsze pozycje.')
                                for pos, value in enumerate(q):
                                    traces.append(dict(case=case, contrast='A_vs_' + condition,
                                                       question=question, state=state, token=pos,
                                                       sensitivity=float(value)))
                    print(f"{case}: I={b['binding_interaction']:.4f} C={b['common_context_effect']:.4f}", flush=True)
    return cases, rejected, traces


def run_nulls(exp, args):
    pair = PAIRS[args.nulls_pair]
    template = TEMPLATES[args.nulls_template]
    rows = []
    generator = torch.Generator(device=exp.device).manual_seed(args.seed)
    for fillers in args.null_fillers:
        for key_first in [True, False]:
            ctx = contexts(template, pair, fillers, key_first)
            reg, reason = regions(exp.tok, ctx)
            if reg is None:
                raise ValueError(f'Konfiguracja kalibracyjna nie jest wyrownana: {reason}')
            margins = {c: [exp.run(t)[0] for t in v[1]] for c, v in ctx.items()}
            case = f'{args.nulls_template}|{args.nulls_pair}|f{fillers}|{"key" if key_first else "cup"}'
            donor_state = {c: [exp.run(t, capture=True)[1] for t in v[1]] for c, v in ctx.items()}
            for donor, recipient in [('A', 'B'), ('B', 'A')]:
                sign = 1 if donor == 'A' else -1
                for layer in args.layers:
                    position = reg['fact_colors']
                    shifts = {'donor': {}, 'null': defaultdict(list)}
                    for target, question in enumerate(['key', 'cup']):
                        source = donor_state[donor][target][layer + 1][position]
                        reference = donor_state[recipient][target][layer + 1][position]
                        delta = source - reference
                        score, _ = exp.run(ctx[recipient][1][target], patch=(layer, position, source))
                        shifts['donor'][question] = score - margins[recipient][target]
                        for _ in range(args.null_draws):
                            permutation = torch.randperm(delta.shape[-1], generator=generator,
                                                         device=exp.device)
                            null = reference + delta[:, permutation]
                            score, _ = exp.run(ctx[recipient][1][target], patch=(layer, position, null))
                            shifts['null'][question].append(score - margins[recipient][target])
                    donor_selective = sign * (shifts['donor']['key'] - shifts['donor']['cup']) / 2
                    null_selective = [sign * (k - c) / 2 for k, c in
                                      zip(shifts['null']['key'], shifts['null']['cup'])]
                    exceed = int(sum(abs(v) >= abs(donor_selective) for v in null_selective))
                    rows.append(dict(case=case, direction=f'{donor}_to_{recipient}', layer=layer,
                                     donor_selective=float(donor_selective),
                                     null_mean=float(np.mean(null_selective)),
                                     null_sd=float(np.std(null_selective)),
                                     null_min=float(np.min(null_selective)),
                                     null_max=float(np.max(null_selective)),
                                     null_same_sign=int(sum(np.sign(v) == np.sign(donor_selective)
                                                            for v in null_selective)),
                                     draws=int(args.null_draws), exceed=exceed,
                                     p_empirical=float((1 + exceed) / (args.null_draws + 1))))
            print(f'{case}: done', flush=True)
    return rows


def plots(out, cases, nulls):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if cases:
        fig, ax = plt.subplots(figsize=(12, 5))
        names = sorted({c['template'] for c in cases})
        pairs = sorted({c['pair'] for c in cases})
        width = .8 / len(pairs)
        for j, pair in enumerate(pairs):
            means = [np.mean([c['binding_interaction'] for c in cases
                              if c['template'] == t and c['pair'] == pair]) for t in names]
            ax.bar(np.arange(len(names)) + (j - (len(pairs) - 1) / 2) * width, means, width, label=pair)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_xticks(range(len(names)), names)
        ax.set(ylabel='mean binding interaction I', title='Replication across templates and noun pairs')
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / 'replication.png', dpi=130); plt.close(fig)
    if nulls:
        fig, ax = plt.subplots(figsize=(9, 5))
        layers = sorted({r['layer'] for r in nulls})
        donor = [np.mean([r['donor_selective'] for r in nulls if r['layer'] == x]) for x in layers]
        low = [np.mean([r['null_min'] for r in nulls if r['layer'] == x]) for x in layers]
        high = [np.mean([r['null_max'] for r in nulls if r['layer'] == x]) for x in layers]
        spread = [np.mean([r['null_sd'] for r in nulls if r['layer'] == x]) for x in layers]
        ax.fill_between(layers, low, high, color='tab:orange', alpha=.18,
                        label='range of 10 null draws (mean per block)')
        ax.errorbar(layers, donor, yerr=spread, fmt='o-', color='tab:blue',
                    label='donor selective shift (error bar = mean null sd)')
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set(xlabel='sampled block', ylabel='selective shift',
               title='Donor effect lies inside the null spread')
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / 'nulls.png', dpi=130); plt.close(fig)


def main():
    global TEMPLATES, PAIRS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='PiotrSty/gollem-v6-250m-sft-answers')
    p.add_argument('--revision', default=None,
                   help='rewizja Hub; domyslnie przypieta dla modelu z badania, a dla innych najnowsza')
    p.add_argument('--out', type=Path, default=Path('runs/replicate'))
    p.add_argument('--part', choices=['replicate', 'nulls', 'both', 'plot'], default='replicate')
    p.add_argument('--templates', nargs='+', choices=list(TEMPLATES), default=list(TEMPLATES))
    p.add_argument('--pairs', nargs='+', choices=list(PAIRS), default=list(PAIRS))
    p.add_argument('--fillers', type=int, nargs='+', default=[0, 4],
                   help='dlugosci wstawki w czesci replicate')
    p.add_argument('--null-fillers', type=int, nargs='+', default=[0, 2, 6],
                   help='dlugosci wstawki w kalibracji null; domyslnie pierwotny zestaw')
    p.add_argument('--layers', type=int, nargs='+', default=[0, 5, 10, 15])
    p.add_argument('--nulls-template', default='T1', choices=list(TEMPLATES))
    p.add_argument('--nulls-pair', default='kluczyk/kubek', choices=list(PAIRS))
    p.add_argument('--null-draws', type=int, default=10)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261027)
    args = p.parse_args()
    if args.part == 'plot':
        summary = json.loads((args.out / 'summary.json').read_text(encoding='utf-8'))
        recorded = []
        if (args.out / 'nulls.csv').exists():
            with (args.out / 'nulls.csv').open(encoding='utf-8', newline='') as f:
                recorded = list(csv.DictReader(f))
                for row in recorded:
                    for key in ('layer', 'draws', 'exceed'):
                        row[key] = int(row[key])
                    for key in ('donor_selective', 'null_mean', 'null_sd', 'null_min', 'null_max', 'p_empirical'):
                        row[key] = float(row[key])
        plots(args.out, summary['cases'], recorded)
        print(f'Redrew figures from recorded data in {args.out.resolve()}', flush=True)
        return
    if args.threads < 1 or len(set(args.fillers)) != len(args.fillers) or any(n < 0 for n in args.fillers):
        p.error('Threads must be positive; fillers distinct and nonnegative.')
    if args.null_draws < 2:
        p.error('Need at least two null draws for a distribution.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    revision = args.revision or (REVISION if args.model == 'PiotrSty/gollem-v6-250m-sft-answers' else None)
    path = str(Path(args.model).resolve()) if Path(args.model).is_dir() else snapshot_download(
        args.model, revision=revision,
        allow_patterns=['config.json', 'tokenizer.json', 'model.safetensors', 'modeling_gollem_v6.py'])
    model, tok = _import_gollem(path).load_gollem_v6(path, args.device)
    exp = Experiment(model, tok, args.device)
    TEMPLATES = {k: v for k, v in TEMPLATES.items() if k in args.templates}
    PAIRS = {k: v for k, v in PAIRS.items() if k in args.pairs}
    args.out.mkdir(parents=True, exist_ok=True)
    cases, rejected, traces, nulls = [], [], [], []
    if args.part in ('replicate', 'both'):
        cases, rejected, traces = run_replicate(exp, args)
    if args.part in ('nulls', 'both'):
        nulls = run_nulls(exp, args)
    summary = dict(model=args.model, requested_revision=revision, resolved_snapshot=path,
                   torch=torch.__version__, device=args.device, threads=args.threads, seed=args.seed,
                   templates=list(TEMPLATES), pairs=list(PAIRS), fillers=args.fillers,
                   null_layers=args.layers, null_draws=args.null_draws,
                   neutral=NEUTRAL, cases=cases, rejected=rejected, nulls=nulls,
                   limitations=['Balanced masculine noun pairs only; token balance is mechanical.',
                     'Template set fixed before measurement; rejections reported, not silently dropped.',
                     'Replication uses one checkpoint and contrasts two forced answers.',
                     'Null p-values use a small number of draws; minimum p is 1/(draws+1).',
                     'Positive selective transfer is not proof of semantic understanding.'])
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if cases:
        write_csv(args.out / 'cases.csv', [{k: v for k, v in c.items()
                                            if k not in ('prompts', 'prompt_ids', 'margins', 'regions')}
                                           for c in cases])
        write_csv(args.out / 'traces.csv', traces)
    if nulls:
        write_csv(args.out / 'nulls.csv', nulls)
    plots(args.out, cases, nulls)
    positives = sum(c['binding_interaction'] > 0 for c in cases)
    print(f'cases={len(cases)} rejected={len(rejected)} positive_I={positives} '
          f'null_rows={len(nulls)} output={args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
