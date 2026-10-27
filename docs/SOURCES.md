# Źródła i zakres ich wykorzystania

## Sprawdzone źródła pierwotne

1. **Yiping Lu i in. — Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View (2019).**
   https://arxiv.org/abs/1906.02762
   Interpretacja dynamiki konwekcji–dyfuzji i schematów Lie–Trottera / Stranga–Marchuka; architektura Macaron FFN–attention–FFN. Uzasadnia szukanie analogii transportowej, nie konkretne utożsamienie CE z ciśnieniem.

2. **Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet — A mathematical perspective on Transformers (2023).**
   https://arxiv.org/abs/2312.10794
   Przeczytana wersja: https://arxiv.org/html/2312.10794v2
   Model oddziałujących cząstek, równanie ciągłości (3.4), klasteryzacja. Główna analiza dotyczy uproszczeń self-attention i normalizacji, często stałych Q/K/V. Energia interakcji (3.5) może rosnąć w atrakcyjnym modelu; nie jest to energia Bernoulliego. Nie przenosimy twierdzeń automatycznie na pełny causal multihead pre-norm LLM.

3. **Samira Abnar, Willem Zuidema — Quantifying Attention Flow in Transformers (ACL 2020).**
   https://aclanthology.org/2020.acl-main.385/
   Attention rollout i attention flow jako przybliżenia mieszania wpływów między tokenami. Nie zastępują przyczynowego testowania wpływu na wynik.

4. **Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov — Locating and Editing Factual Associations in GPT (2022).**
   https://rome.baulab.info/
   https://arxiv.org/abs/2202.05262
   Causal tracing przez zakłócanie obliczenia i przywracanie aktywacji. Wykorzystujemy zasadę interwencji, nie implementujemy edycji wag ROME. Patching nie jest nową metodą tego projektu.

5. **Sarthak Jain, Byron C. Wallace — Attention is not Explanation (NAACL 2019).**
   https://aclanthology.org/N19-1357/
   Ostrzeżenie przed utożsamianiem wag uwagi z przyczynowym znaczeniem. Ważna granica: eksperymenty pracy dotyczą przede wszystkim modeli z enkoderami BiRNN, nie są ogólnym twierdzeniem o każdym transformerze.

## Model i dane

- **PiotrSty/gollem-v6-250m-sft-answers:** https://huggingface.co/PiotrSty/gollem-v6-250m-sft-answers
- **Kod checkpointa:** https://huggingface.co/PiotrSty/gollem-v6-250m-sft-answers/blob/3a60edc280d93fb73ead0f5a5ba220986525aa76/modeling_gollem_v6.py
- **Rewizja z pomiarów:** `3a60edc280d93fb73ead0f5a5ba220986525aa76`.
- Model jest natywnym modułem PyTorch, nie modelem `AutoModelForCausalLM` z `auto_map`. Własny loader, tokenizer `tokenizers.Tokenizer`, bloki zwracające `(x, v0)`, końcowa norma `lnf` i tied `head`.
- Karta modelu deklaruje CC-BY-SA-4.0 i pochodzenie od SlayerLab/GoLLeM-v6-250M. Wagi ani kod modelu nie są kopiowane do tego repozytorium; pobiera je loader.
- **Korpus pierwotnego pomiaru:** https://huggingface.co/datasets/PiotrSty/wolne-lektury-polish-literature-corpus
  Rewizji korpusu nie zapisano. Surowe teksty książek nie są publikowane w tym repozytorium.

## Wątki z początkowej rozmowy, których nie używamy jako potwierdzonej podstawy

W rozmowie pojawiły się również Transformer Dynamics (arXiv:2502.12131), Wavy Transformer (arXiv:2508.12787), GRAND (arXiv:2106.10934), przypisanie DIFFormer do arXiv:2409.09111 i praca wskazana jako arXiv:2412.18288. Ten zapis zachowuje historię tropów, **nie potwierdza poprawności wszystkich tytułów, identyfikatorów i wcześniejszych streszczeń**. W późniejszym badaniu bezpośrednio sprawdzono źródła z listy powyżej. Nie opieramy wniosków na niezweryfikowanych przypisaniach z początkowej rozmowy.

Nie znaleziono w tym badaniu potwierdzonego źródła wyprowadzającego prawo Bernoulliego lub efekt Venturiego dla pełnych warstw LLM. To nie jest dowód braku takiej literatury.
