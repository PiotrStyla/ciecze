#!/usr/bin/env python3
"""
Test analogii Bernoulli / Venturi dla warstw transformera.

Dla kazdego stanu l strumienia rezydualnego h_l (l=0 to embedding, l=L to
wyjscie ostatniego bloku):
  v_l  "predkosc"  = srednia ||h_{l+1} - h_l|| po tokenach
  n_l  norma       = srednia ||h_l||
  A_l  "przekroj"  = participation ratio widma kowariancji tokenow (efektywny
                     wymiar), liczony osobno dla kazdego okna i usredniany
  p_l  "cisnienie" = strata CE logit lens (next-token) ze stanu h_l

Testy:
  1. Ciaglosc:  v*A ~ const ?   CV(v*A) vs CV(v), CV(A)
  2. Venturi:   log v ~ log A, nachylenie (przewidywane -1)
  3. Bernoulli: v^2/2 = C - k*p  (k>0 i wysokie R^2 = zgodnosc)

Kazdy test ma test permutacyjny (p-value), bo przy ~20 warstwach samo R^2
niewiele znaczy, i wersje po usunieciu trendu liniowego z glebokoscia:
dwie dowolne monotonne krzywe wzdluz warstw daja R^2 ~ 1 bez zwiazku.
Wersja "wzgledna" liczy v = ||dh||/||h||, bo strumien rezydualny rosnie
w normie o rzedy wielkosci i predkosc bezwzgledna to glownie ten wzrost.

Uruchomienie (PowerShell):
  pip install torch transformers numpy matplotlib datasets
  # wlasny korpus, jeden tekst na linie:
  python bernoulli_layers.py --model PiotrSty/gollem-v6-250m-sft-answers --file teksty.txt
  # korpus prosto z Hub:
  python bernoulli_layers.py --dataset PiotrSty/wolne-lektury-polish-literature-corpus
  # dowolny model HF z hidden_states:
  python bernoulli_layers.py --model gpt2 --backend hf --file teksty.txt

Backendy:
  gollem - natywny kod GoLLeM (modeling_gollem_v6.py: hooki na tok + blocks,
           koncowa norma lnf, glowa tied head)
  hf     - AutoModelForCausalLM + output_hidden_states (Llama, GPT-2, NeoX...)
  auto   - gollem, jesli config ma n_embd/n_layer, inaczej hf
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TEXTS = [
    "Kraków jest jednym z najstarszych miast w Polsce i dawną stolicą kraju.",
    "Równanie Bernoulliego opisuje zachowanie cieczy nieściśliwej w przepływie ustalonym.",
    "Modele językowe przetwarzają tekst warstwa po warstwie, aktualizując reprezentację każdego tokenu.",
    "Wczoraj wieczorem padał deszcz, więc zostaliśmy w domu i czytaliśmy książki.",
    "Rura o zmiennym przekroju powoduje wzrost prędkości przepływu w przewężeniu.",
    "Nauka języków obcych wymaga regularnej praktyki, cierpliwości i dobrych materiałów.",
    "Wisła przepływa przez Kraków i uchodzi do Morza Bałtyckiego koło Gdańska.",
    "Sztuczna inteligencja zmienia sposób, w jaki pracujemy, uczymy się i komunikujemy.",
]


# ---------------------------------------------------------------- modele

def find_final_norm(model):
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm",
                 "model.decoder.final_layer_norm", "transformer.norm_f",
                 "model.final_layernorm", "lnf", "norm"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
        except AttributeError:
            continue
        if obj is not None:
            return obj
    return None


class HFBackend:
    """AutoModelForCausalLM: stany z output_hidden_states (ostatni juz z norma koncowa)."""

    def __init__(self, model_id, device, trust_remote_code=False):
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        kw = dict(trust_remote_code=trust_remote_code)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, **kw)
        except TypeError:  # transformers < 4.56
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, **kw)
        self.model.to(device).eval()
        self.final_norm = find_final_norm(self.model)
        self.lm_head = self.model.get_output_embeddings()
        self.last_normed = self.final_norm is not None
        if self.final_norm is None:
            print("Ostrzezenie: nie znaleziono koncowej normy; logit lens bez normalizacji.")

    def encode(self, text):
        return self.tok(text)["input_ids"]

    @torch.no_grad()
    def states(self, ids):
        out = self.model(ids, output_hidden_states=True)
        return [h[0].float() for h in out.hidden_states]


class GollemBackend:
    """Natywny GoLLeM: hooki na osadzaniu i blokach, lnf + tied head."""

    def __init__(self, model_id, device):
        from tokenizers import Tokenizer
        path = _local_dir(model_id, ["config.json", "model.safetensors", "tokenizer.json",
                                     "modeling_gollem_v6.py"])
        mod = _import_gollem(path)
        model, tok = mod.load_gollem_v6(path, device)
        self.model = model
        self.tok = tok
        self.final_norm = getattr(model, "lnf", None)
        self.lm_head = model.head
        self.last_normed = False
        self._buf = []
        modules = [model.tok] + list(model.blocks)
        for m in modules:
            m.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        x = output[0] if isinstance(output, (tuple, list)) else output
        self._buf.append(x)

    def encode(self, text):
        return self.tok.encode(text).ids

    @torch.no_grad()
    def states(self, ids):
        self._buf = []
        self.model(ids)
        return [h[0].float() for h in self._buf]


def _gollem_cfg(path):
    import json
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _import_gollem(path):
    f = os.path.join(path, "modeling_gollem_v6.py")
    if not os.path.exists(f):
        sys.exit(f"Brak {f} - to nie jest checkpoint GoLLeM v6.")
    if path not in sys.path:
        sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("modeling_gollem_v6", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modeling_gollem_v6"] = mod
    spec.loader.exec_module(mod)
    return mod


def _local_dir(model_id, patterns):
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=patterns)


def is_gollem(model_id):
    import json
    try:
        if os.path.isdir(model_id):
            p = os.path.join(model_id, "config.json")
            if not os.path.exists(p):
                return False
            cfg = json.load(open(p, encoding="utf-8"))
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(model_id, "config.json")
            cfg = json.load(open(p, encoding="utf-8"))
    except Exception:
        return False
    return "n_embd" in cfg and "n_layer" in cfg


def build_backend(model_id, backend, device, trust_remote_code):
    if backend == "auto":
        backend = "gollem" if is_gollem(model_id) else "hf"
    if backend == "gollem":
        return GollemBackend(model_id, device)
    return HFBackend(model_id, device, trust_remote_code)


# ---------------------------------------------------------------- teksty

def read_file(path):
    with open(path, encoding="utf-8") as f:
        return [t.strip() for t in f if t.strip()]


def read_dataset(spec, n):
    """spec = repo | repo:split | repo:split:kolumna"""
    parts = spec.split(":")
    repo = parts[0]
    split = parts[1] if len(parts) > 1 and parts[1] else "train"
    col = parts[2] if len(parts) > 2 and parts[2] else None
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    rows = list(ds.take(n))
    if not rows:
        sys.exit(f"Pusty split {repo}:{split}")
    if col is None:
        cands = [k for k, v in rows[0].items()
                 if isinstance(v, str) and np.mean([len(r.get(k) or "") for r in rows]) > 200]
        if not cands:
            sys.exit(f"Nie znalazlem dlugiego pola tekstowego; kolumny: {list(rows[0])}. Podaj repo:split:kolumna")
        col = cands[0]
    print(f"Dane: {repo}:{split}:{col} ({len(rows)} rekordow)")
    return [r[col].strip() for r in rows if (r.get(col) or "").strip()]


def chunk_ids(ids, max_len, max_chunks=4):
    """Okna o stalej dlugosci (staly sufit dla PR). Dlugi tekst -> kilka rowno rozlozonych okien."""
    if len(ids) <= max_len:
        return [ids]
    starts = np.linspace(0, len(ids) - max_len, num=min(max_chunks, 1 + (len(ids) - max_len) // max_len), dtype=int)
    return [ids[s:s + max_len] for s in starts]


# ---------------------------------------------------------------- miary

def participation_ratio(X):
    """PR widma kowariancji (X: [n, d], n>1). PR <= n, wiec dlugosc okna ogranicza A.
    Liczone w float64: strumien rezydualny ma normy ~1e4-1e5 i float32 gubi wariancje. """
    if X.shape[0] < 2:
        return float("nan")
    X = X.double()
    X = X - X.mean(0, keepdim=True)
    lam = torch.linalg.svdvals(X) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def fit_line(x, y):
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, icpt, r2


def detrend(y):
    """Odejmij trend liniowy po glebokosci - inaczej dwie monotonne krzywe zawsze dadza R2~1."""
    x = np.arange(len(y), dtype=float)
    slope, icpt, _ = fit_line(x, y)
    return y - (slope * x + icpt)


def perm_pvalue(x, y, n_perm=5000, seed=0):
    """p-value dla R^2 dopasowania liniowego: permutujemy y."""
    _, _, r2_obs = fit_line(x, y)
    if not np.isfinite(r2_obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        _, _, r2 = fit_line(x, rng.permutation(y))
        hits += r2 >= r2_obs
    return (hits + 1) / (n_perm + 1)


# ---------------------------------------------------------------- glowna petla

def measure(backend, texts, device, max_len, skip, max_chunks=4, max_windows=256):
    vel = relvel = norm = ce = pr = None
    tok_count = ce_count = pr_count = 0
    n_states = None
    windows = 0

    with torch.no_grad():
        for text in texts:
            if windows >= max_windows:
                break
            for ids_list in chunk_ids(backend.encode(text), max_len, max_chunks):
                if windows >= max_windows:
                    break
                if len(ids_list) < skip + 3:
                    continue
                ids = torch.tensor([ids_list], device=device)
                T = ids.shape[1]
                states = backend.states(ids)
                if n_states is None:
                    n_states = len(states)
                    vel = np.zeros(n_states - 1)
                    relvel = np.zeros(n_states - 1)
                    norm = np.zeros(n_states)
                    ce = np.zeros(n_states)
                    pr = np.zeros(n_states)
                windows += 1
                for l in range(n_states):
                    h = states[l][skip:]
                    norm[l] += h.norm(dim=-1).sum().item()
                    pr[l] += participation_ratio(h)
                    pr_count += 1
                    if l < n_states - 1:
                        d = (states[l + 1] - states[l])[skip:]
                        dn = d.norm(dim=-1)
                        vel[l] += dn.sum().item()
                        relvel[l] += (dn / h.norm(dim=-1).clamp_min(1e-8)).sum().item()
                    x = states[l]
                    if not (backend.last_normed and l == n_states - 1) and backend.final_norm is not None:
                        x = backend.final_norm(x)
                    logits = backend.lm_head(x[skip:T - 1])
                    ce[l] += torch.nn.functional.cross_entropy(
                        logits, ids[0, skip + 1:T], reduction="sum").item()
                tok_count += T - skip
                ce_count += T - 1 - skip

    return dict(
        v=vel / tok_count,
        v_rel=relvel / tok_count,
        norm=norm / tok_count,
        p=ce / ce_count,
        A=pr / max(pr_count // n_states, 1),
        windows=windows,
        tokens=tok_count,
        states=n_states,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="PiotrSty/gollem-v6-250m-sft-answers",
                    help="id z Hub albo katalog lokalny")
    ap.add_argument("--backend", choices=["auto", "gollem", "hf"], default="auto")
    ap.add_argument("--file", help="plik tekstowy, jeden tekst na linie")
    ap.add_argument("--dataset", help="repo[:split[:kolumna]] z HF Hub")
    ap.add_argument("--max-texts", type=int, default=64, help="ile rekordow wziac z --dataset")
    ap.add_argument("--max-len", type=int, default=256, help="dlugosc okna w tokenach")
    ap.add_argument("--max-chunks", type=int, default=4, help="ile okien z jednego dlugiego tekstu")
    ap.add_argument("--max-windows", type=int, default=256, help="laczny budzet okien (czas obliczen)")
    ap.add_argument("--out", default="bernoulli_out")
    ap.add_argument("--keep-first", action="store_true",
                    help="nie pomijaj pierwszego tokenu (attention sink ma ogromna norme)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--perm", type=int, default=5000, help="liczba permutacji do p-value")
    args = ap.parse_args()

    if args.file:
        texts = read_file(args.file)
    elif args.dataset:
        texts = read_dataset(args.dataset, args.max_texts)
    else:
        texts = DEFAULT_TEXTS
        print("Uwaga: maly domyslny zestaw; podaj --file albo --dataset dla stabilnych wynikow.")

    backend = build_backend(args.model, args.backend, args.device, args.trust_remote_code)
    skip = 0 if args.keep_first else 1
    m = measure(backend, texts, args.device, args.max_len, skip, args.max_chunks, args.max_windows)

    L = m["states"] - 1
    v, v_rel, norm, p, A = m["v"], m["v_rel"], m["norm"], m["p"], m["A"]
    AA, pp, nn = A[:L], p[:L], norm[:L]

    def tests(vv):
        cont = vv * AA
        s_ven, _, r2_ven = fit_line(np.log(AA), np.log(vv))
        slope_b, icpt_b, r2_b = fit_line(pp, vv ** 2 / 2)
        # to samo po usunieciu wspolnego trendu z glebokoscia
        lv, la = detrend(np.log(vv)), detrend(np.log(AA))
        s_ven_d, _, r2_ven_d = fit_line(la, lv)
        slope_bd, _, r2_bd = fit_line(detrend(pp), detrend(vv ** 2 / 2))
        return dict(
            cont=cont,
            cv_cont=cont.std() / cont.mean(), cv_v=vv.std() / vv.mean(), cv_A=AA.std() / AA.mean(),
            s_ven=s_ven, r2_ven=r2_ven, p_ven=perm_pvalue(np.log(AA), np.log(vv), args.perm),
            s_ven_d=s_ven_d, r2_ven_d=r2_ven_d, p_ven_d=perm_pvalue(la, lv, args.perm),
            corr_vA=np.corrcoef(vv, AA)[0, 1],
            slope_b=slope_b, icpt_b=icpt_b, r2_b=r2_b, k=-slope_b,
            p_ber=perm_pvalue(pp, vv ** 2 / 2, args.perm),
            k_d=-slope_bd, r2_b_d=r2_bd,
            p_ber_d=perm_pvalue(detrend(pp), detrend(vv ** 2 / 2), args.perm),
            corr_vp=np.corrcoef(vv ** 2, pp)[0, 1],
        )

    tab = tests(v)          # predkosc bezwzgledna
    tab_rel = tests(v_rel)  # predkosc wzgledna: ||dh|| / ||h||
    cont = tab["cont"]

    os.makedirs(args.out, exist_ok=True)
    csv = os.path.join(args.out, "layers.csv")
    with open(csv, "w", encoding="utf-8") as f:
        f.write("layer,v,v_rel,norm,A_pr,p_ce,v_times_A\n")
        for l in range(L):
            f.write(f"{l},{v[l]:.6f},{v_rel[l]:.6f},{norm[l]:.6f},{A[l]:.4f},{p[l]:.6f},{cont[l]:.6f}\n")

    print(f"\nModel: {args.model} ({type(backend).__name__}) | warstw: {L} | okien: {m['windows']} "
          f"| tokenow: {m['tokens']} | bez pierwszego tokenu: {not args.keep_first}")
    print(f"{'l':>3} {'v':>10} {'v_rel':>8} {'norm':>10} {'A (PR)':>9} {'p (CE)':>8} {'v*A':>10}")
    for l in range(L):
        print(f"{l:>3} {v[l]:>10.3f} {v_rel[l]:>8.3f} {norm[l]:>10.2f} {A[l]:>9.2f} {p[l]:>8.3f} {cont[l]:>10.2f}")

    for label, t in (("v = ||dh|| (bezwzgledna)", tab), ("v = ||dh||/||h|| (wzgledna)", tab_rel)):
        print(f"\n--- Testy: {label} ---")
        print(f"Ciaglosc  : CV(v*A) = {t['cv_cont']:.3f}   (CV v = {t['cv_v']:.3f}, CV A = {t['cv_A']:.3f}); "
              f"hipoteze wspiera CV(v*A) wyraznie mniejsze od obu")
        print(f"Venturi   : nachylenie log v ~ log A = {t['s_ven']:.2f} (przewidywane -1), R2 = {t['r2_ven']:.2f}, "
              f"p_perm = {t['p_ven']:.4f}, corr(v,A) = {t['corr_vA']:.2f}")
        print(f"Bernoulli : v^2/2 = C - k*p -> k = {t['k']:.4g}, R2 = {t['r2_b']:.2f}, p_perm = {t['p_ber']:.4f}, "
              f"corr(v^2,p) = {t['corr_vp']:.2f} (zgodnosc: k>0 i istotne R2)")
        print(f"  po usunieciu trendu z glebokoscia: Venturi nachylenie = {t['s_ven_d']:.2f}, R2 = {t['r2_ven_d']:.2f}, "
              f"p_perm = {t['p_ven_d']:.4f} | Bernoulli k = {t['k_d']:.4g}, R2 = {t['r2_b_d']:.2f}, "
              f"p_perm = {t['p_ber_d']:.4f}")

    tol = 0.9 * (args.max_len - skip)
    if np.nanmax(A) > tol:
        print(f"\nUwaga: A dochodzi do {np.nanmax(A):.1f}, a PR jest ograniczone liczba tokenow w oknie "
              f"({args.max_len - skip}). Zwieksz --max-len, zeby przekroj nie byl uciety sufitem.")
    print(f"\nZapisano: {csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(14, 7))
        xs = np.arange(L)
        for a, (y, t) in zip(ax.flat[:5], [(v, "v (predkosc)"), (nn, "||h|| (norma)"),
                                           (AA, "A (przekroj, PR)"), (pp, "p (CE logit lens)"),
                                           (cont, "v*A")]):
            a.plot(xs, y, "o-")
            a.set_title(t)
            a.set_xlabel("warstwa")
        ax.flat[5].scatter(pp, v ** 2 / 2)
        ax.flat[5].plot(pp, tab["slope_b"] * pp + tab["icpt_b"], "r-")
        ax.flat[5].set_title(f"Bernoulli: v^2/2 vs p (R2={tab['r2_b']:.2f})")
        ax.flat[5].set_xlabel("p")
        plt.tight_layout()
        png = os.path.join(args.out, "layers.png")
        plt.savefig(png, dpi=140)
        print(f"Wykres: {png}")
    except ImportError:
        print("Brak matplotlib - pominieto wykres.")


if __name__ == "__main__":
    main()
