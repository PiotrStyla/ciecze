#!/usr/bin/env python3
"""Odczyt faktu czy preferencja slow? Kontekst x pytanie w GoLLeM v6.

python binding_flow.py --baseline-only --out runs/binding_baseline
python binding_flow.py --out runs/binding
Smoke: --fillers 0 --layers 0 19 --out runs/binding_smoke

Uzywa Experiment z dye_flow.py i konstrukcji par z valve_flow.py. Nie zmienia
wag ani poprzednich pomiarow. Domyslnie przypina oryginalna rewizje modelu.

Warunki: A = czerwony kluczyk / niebieski kubek, B = odwrotne przypisanie,
N = oba przedmioty 'opisany' (bez podanej barwy, tez po 2 tokeny). W kazdym
warunku oba pytania maja ten sam koncowy prefiks odpowiedzi. Wszystkie
pozycje sa wyrownane; neutralnego opisu NIE oceniamy jako poprawny/bledny.

m = log P(pelne czerwony) - log P(pelne niebieski), teacher forcing.
R_key = m(A,key)-m(B,key); R_cup = m(A,cup)-m(B,cup).
I = (R_key-R_cup)/2: interakcja przypisania kolorow i celu pytania.
C = (R_key+R_cup)/2: wspolny efekt zmiany opisu.
Addytywna stala preferencja koloru i addytywny efekt nazwy pytania znosza
sie w I. Dodatnie I nie dowodzi semantycznego rozumienia, nie jest p-value
ani trafnoscia odpowiedzi. Kontrola N ma inne znaczenie i tez moze wywolywac
skojarzenia; nie jest czystym, bezwarunkowym priorem modelu.

Patching przenosi stany miedzy A i B przy NIEZMIENIONYM pytaniu. Regiony:
oba kolory faktow razem, nazwa w pytaniu, odczyt. Oba kierunki; norm-matched
null; self-patching. Z bocznego value_residual v0 pozostaje stan biorcy.
Efekt selektywny patchingu to polowa roznicy przesuniec dla pytania o kluczyk
i kubek, ze znakiem dodatnim dla kierunku A->B i ujemnym dla B->A.
Jedno losowanie null na interwencje, bez twierdzen o istotnosci.

Artefakty: summary.json, baselines.csv, traces.csv; przy pelnym przebiegu
patches.csv, transfers.csv, overview.png i transfer.png. Kontrole niewykonane
w trybie baseline-only maja wartosc null, nie zero.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from bernoulli_layers import _import_gollem
from dye_flow import Experiment, write_csv
from valve_flow import make_pair, positions

REVISION = '3a60edc280d93fb73ead0f5a5ba220986525aa76'


def contrasts(margins):
    a, b = margins['A'], margins['B']
    key = a[0] - b[0]
    cup = a[1] - b[1]
    return dict(key_context_response=key, cup_context_response=cup,
                binding_interaction=(key - cup) / 2, common_context_effect=(key + cup) / 2,
                neutral_question_gap=margins['N'][0] - margins['N'][1],
                opposite_context_responses=key > 0 and cup < 0)


def prepare(tok, count, key_first):
    context, a = make_pair(count, key_first, False)
    _, b = make_pair(count, key_first, True)
    neutral = [text.replace('czerwony', 'opisany').replace('niebieski', 'opisany') for text in a]
    texts = dict(A=a, B=b, N=neutral)
    ids = {condition: [tok.encode(t).ids for t in pair] for condition, pair in texts.items()}
    if len({len(row) for pair in ids.values() for row in pair}) != 1:
        raise ValueError('Warunki kontekstu nie sa wyrownane tokenowo.')
    _, old_regions = positions(tok, context, a)
    regions = dict(fact_colors=old_regions['facts_control'],
                   question_subject=old_regions['question_subject'], readout=old_regions['readout'])
    for target in range(2):
        for condition in ['B', 'N']:
            changed = [i for i, (x, y) in enumerate(zip(ids['A'][target], ids[condition][target])) if x != y]
            if changed != regions['fact_colors']:
                raise ValueError('Zmiana kontekstu nie ogranicza sie do kolorow.')
    return texts, ids, regions


def measure_case(exp, case, count, key_first, layers, seed, baseline_only):
    texts, ids, regions = prepare(exp.tok, count, key_first)
    runs = {condition: [exp.run(t, capture=True) for t in pair] for condition, pair in texts.items()}
    margins = {condition: [r[0] for r in pair] for condition, pair in runs.items()}
    b = dict(case=case, fillers=count, key_first=key_first, prompts=texts, prompt_ids=ids,
             regions=regions, margins=margins, **contrasts(margins),
             max_earlier_relative_difference=0.0, max_self_patch_error=None,
             max_null_norm_relative_error=None)
    baselines, traces, patches = [], [], []
    for condition, pair in texts.items():
        for target in range(2):
            expected = None if condition == 'N' else (1 if (condition == 'A') == (target == 0) else -1)
            baselines.append(dict(case=case, condition=condition, question=['key', 'cup'][target],
                prompt=pair[target], margin_red_minus_blue=margins[condition][target],
                correct_preference=None if expected is None else margins[condition][target] * expected > 0))
    for condition in ['B', 'N']:
        for target in range(2):
            for state, (a, other) in enumerate(zip(runs['A'][target][1], runs[condition][target][1])):
                q = ((a-other).norm(dim=-1) /
                     ((a.norm(dim=-1)+other.norm(dim=-1))/2).clamp_min(1e-8)).cpu().numpy()
                error = float(q[:min(regions['fact_colors'])].max())
                b['max_earlier_relative_difference'] = max(b['max_earlier_relative_difference'], error)
                if error > 1e-5:
                    raise AssertionError('Zmiana opisu zmienia pozycje przed pierwszym kolorem.')
                for pos, value in enumerate(q):
                    traces.append(dict(case=case, contrast='A_vs_'+condition, question=['key','cup'][target],
                                       state=state, token=pos, sensitivity=float(value)))
    if baseline_only:
        return b, baselines, traces, patches
    b['max_self_patch_error'] = b['max_null_norm_relative_error'] = 0.0
    generator = torch.Generator(device=exp.device).manual_seed(seed)
    for donor, recipient in [('A','B'), ('B','A')]:
        for target in range(2):
            for layer in layers:
                for region, pos in regions.items():
                    source = runs[donor][target][1][layer+1][pos]
                    reference = runs[recipient][target][1][layer+1][pos]
                    delta = source-reference
                    permutation = torch.randperm(delta.shape[-1], generator=generator, device=exp.device)
                    null = reference+delta[:,permutation]
                    original_norm = delta.norm(dim=-1)
                    norm_error = float(((original_norm-(null-reference).norm(dim=-1)).abs() /
                                        original_norm.clamp_min(1e-8)).max())
                    b['max_null_norm_relative_error'] = max(b['max_null_norm_relative_error'], norm_error)
                    if norm_error > 1e-4:
                        raise AssertionError('Norma null jest niezgodna z norma dawcy.')
                    for kind, value in [('donor',source), ('norm_matched_null',null)]:
                        score, _ = exp.run(texts[recipient][target], patch=(layer,pos,value))
                        patches.append(dict(case=case, direction=donor+'_to_'+recipient,
                            question=['key','cup'][target], layer=layer, region=region, kind=kind,
                            recipient_margin=margins[recipient][target], donor_margin=margins[donor][target],
                            patched_margin=score, margin_shift=score-margins[recipient][target],
                            perturbation_norm=float((value-reference).norm())))
                pos = regions['readout']
                score, _ = exp.run(texts[recipient][target], patch=(layer,pos,runs[recipient][target][1][layer+1][pos]))
                error = abs(score-margins[recipient][target])
                b['max_self_patch_error'] = max(b['max_self_patch_error'],error)
                if error > 1e-4:
                    raise AssertionError('Self-patching zmienia wynik.')
    return b, baselines, traces, patches


def transfers(patches):
    grouped = defaultdict(dict)
    for row in patches:
        key = tuple(row[k] for k in ['case','direction','layer','region','kind'])
        grouped[key][row['question']] = row['margin_shift']
    rows = []
    for (case,direction,layer,region,kind), shifts in grouped.items():
        if set(shifts) != {'key','cup'}:
            raise ValueError('Brakuje jednego pytania w parze interwencji.')
        sign = 1 if direction == 'A_to_B' else -1
        rows.append(dict(case=case,direction=direction,layer=layer,region=region,kind=kind,
                         selective_shift=sign*(shifts['key']-shifts['cup'])/2,
                         common_shift=sign*(shifts['key']+shifts['cup'])/2))
    return rows


def plots(out, cases, transfer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(cases)); fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].bar(x-.18,[b['key_context_response'] for b in cases],.36,label='question: key')
    ax[0].bar(x+.18,[b['cup_context_response'] for b in cases],.36,label='question: cup')
    ax[0].set(title='Response to swapping fact colors',ylabel='margin(A) - margin(B)')
    ax[1].bar(x-.18,[b['binding_interaction'] for b in cases],.36,label='binding interaction I')
    ax[1].bar(x+.18,[b['common_context_effect'] for b in cases],.36,label='common effect C')
    ax[1].set(title='Selective versus common response',ylabel='log-likelihood margin units')
    for a in ax:
        a.axhline(0,color='gray',linewidth=.8); a.legend()
        a.set_xticks(x,[b['case'] for b in cases],rotation=35,ha='right')
    fig.tight_layout();fig.savefig(out/'overview.png',dpi=130);plt.close(fig)
    if not transfer:
        return
    fig, axes = plt.subplots(1,3,figsize=(14,5))
    for ax, region in zip(axes,['fact_colors','question_subject','readout']):
        layers = sorted({r['layer'] for r in transfer})
        for kind in ['donor','norm_matched_null']:
            means = [np.mean([r['selective_shift'] for r in transfer if r['region']==region
                              and r['layer']==layer and r['kind']==kind]) for layer in layers]
            ax.plot(layers,means,'o-',label=kind)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set(title=region,xlabel='sampled block',ylabel='selective shift; mean over cases/directions')
        ax.legend()
    fig.tight_layout();fig.savefig(out/'transfer.png',dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model',default='PiotrSty/gollem-v6-250m-sft-answers')
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--out',type=Path,default=Path('runs/binding'))
    p.add_argument('--fillers',type=int,nargs='+',default=[0,2,6])
    p.add_argument('--layers',type=int,nargs='+')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--baseline-only',action='store_true')
    args=p.parse_args()
    if args.threads<1 or len(set(args.fillers))!=len(args.fillers) or any(n<0 for n in args.fillers):
        p.error('Threads must be positive; fillers distinct and nonnegative.')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    path=str(Path(args.model).resolve()) if Path(args.model).is_dir() else snapshot_download(
        args.model,revision=args.revision,allow_patterns=['config.json','tokenizer.json','model.safetensors','modeling_gollem_v6.py'])
    model,tok=_import_gollem(path).load_gollem_v6(path,args.device)
    exp=Experiment(model,tok,args.device);n=len(model.blocks)
    layers=args.layers if args.layers is not None else sorted({0,n//4,n//2,3*n//4,n-1})
    if len(set(layers))!=len(layers) or any(i<0 or i>=n for i in layers):
        p.error(f'Layers must be distinct indices in 0..{n-1}.')
    args.out.mkdir(parents=True,exist_ok=True)
    cases=[];baseline_rows=[];trace_rows=[];patch_rows=[]
    for count in args.fillers:
        for key_first in [True,False]:
            case=f"{'key_first' if key_first else 'cup_first'}_f{count}"
            b,bs,ts,ps=measure_case(exp,case,count,key_first,layers,args.seed+len(cases),args.baseline_only)
            cases.append(b);baseline_rows.extend(bs);trace_rows.extend(ts);patch_rows.extend(ps)
            print(f"{case}: R_key={b['key_context_response']:.4f} R_cup={b['cup_context_response']:.4f} "
                  f"I={b['binding_interaction']:.4f} C={b['common_context_effect']:.4f} "
                  f"neutral={b['margins']['N']}",flush=True)
    transfer=transfers(patch_rows)
    summary=dict(model=args.model,requested_revision=args.revision,resolved_snapshot=path,torch=torch.__version__,
                 device=args.device,threads=args.threads,seed=args.seed,sampled_blocks=layers,cases=cases,
                 limitations=['Exploratory: one checkpoint and an existing pilot-selected template.',
                   'Interaction cancels additive biases, not every lexical or positional heuristic.',
                   'Neutral opisany changes meaning; it is not an unconditional model prior.',
                   'Single norm-matched null; no p-values or unique causal-route claims.',
                   'Residual stream is patched; recipient value_residual v0 is unchanged.',
                   'Forced two-answer contrast is not free-generation accuracy.'])
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(args.out/'baselines.csv',baseline_rows);write_csv(args.out/'traces.csv',trace_rows)
    if patch_rows:
        write_csv(args.out/'patches.csv',patch_rows);write_csv(args.out/'transfers.csv',transfer)
    plots(args.out,cases,transfer)
    print(f'Cases={len(cases)} positive_I={sum(b["binding_interaction"]>0 for b in cases)} '
          f'interventions={len(patch_rows)} output={args.out.resolve()}',flush=True)


if __name__=='__main__':
    main()
