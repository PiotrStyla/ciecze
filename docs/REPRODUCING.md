# Odtwarzanie i sprawdzanie wyników

## Środowisko i model

Wersje głównych pakietów zapisano w `requirements.txt`; system, Python i build Torch w `reproducibility/environment.json`. Nie jest to pełny lock wszystkich przechodnich zależności. Instalacji w nowym pustym środowisku nie weryfikowano osobno.

GoLLeM jest ładowany jako natywny model PyTorch. Jego kod pochodzi z checkpointa i jest wykonywany podczas ładowania. Nie wystarczy `AutoModelForCausalLM` ani flaga `--trust-remote-code` dla tej konfiguracji.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('PiotrSty/gollem-v6-250m-sft-answers', revision='3a60edc280d93fb73ead0f5a5ba220986525aa76', local_dir='models/gollem-v6', allow_patterns=['config.json','model.safetensors','tokenizer.json','modeling_gollem_v6.py'])"
```

Pobranie modelu wymaga około 1,1 GB transferu dla samych wag. Nie wymaga Runpoda ani płatnego API. Repozytorium nie zawiera tokenów dostępu. Przy niestandardowej instalacji CPU/GPU sposób instalacji PyTorch może wymagać dostosowania do platformy; plik zależności odzwierciedla zaobserwowane wersje, nie gwarantuje dostępności wheel dla każdej platformy.

## Dwa barwniki

```powershell
python dye_flow.py --model models/gollem-v6 --device cpu --threads 4 --fillers 0 2 6 --layers 0 5 10 15 19 --seed 20261027 --out runs/dye
```

Odpowiada pełnemu przebiegowi w `results/dye`. Skrypt generuje prompty deterministycznie; zapisuje teksty, tokeny, surowe marginesy i kontrole. `pilot.json` jest historycznym wynikiem doboru szablonu i nie jest ponownie tworzony przez główny skrypt.

Krótka próba z sesji:

```powershell
python dye_flow.py --model models/gollem-v6 --fillers 0 --layers 0 19 --out runs/dye_smoke
```

Zachowany smoke pochodzi z wcześniejszej wersji pomocniczego zapisu kontroli norm. Służy jako historia walidacji; pełny przebieg ma kontrolę faktycznej normy zastosowanych zaburzeń.

## Zawór pytania

```powershell
python valve_flow.py --model models/gollem-v6 --device cpu --threads 4 --fillers 0 2 6 --layers 0 5 10 15 19 --seed 20261027 --out runs/valve
```

Można najpierw uruchomić samo sprawdzenie zachowania:

```powershell
python valve_flow.py --model models/gollem-v6 --baseline-only --out runs/valve_baseline
```

Nie dobieraj kolejnych szablonów wyłącznie do poprawienia zapisanych wyników i nie traktuj takiego doboru jako testu niezależnego. W opublikowanym przebiegu pozostawiono wszystkie 12 par, mimo zaledwie 2/12 poprawnych par preferencji.

Krótka próba:

```powershell
python valve_flow.py --model models/gollem-v6 --fillers 0 --layers 0 19 --out runs/valve_smoke
```

## Kontrola treści opisu i wiązania przedmiot–kolor

```powershell
python binding_flow.py --model models/gollem-v6 --baseline-only --out runs/binding_baseline
python binding_flow.py --model models/gollem-v6 --device cpu --threads 4 --fillers 0 2 6 --layers 0 5 10 15 19 --seed 20261027 --out runs/binding
```

Jeśli nie podasz lokalnego katalogu modelu, ten skrypt pobiera domyślnie przypiętą rewizję `3a60edc280d93fb73ead0f5a5ba220986525aa76`, a nie ruchomy `main`.

Krótka próba: `python binding_flow.py --fillers 0 --layers 0 19 --out runs/binding_smoke`.

Zachowano pełny przebieg w `results/binding`, krótką próbę w `results/smoke/binding` i wcześniejszy baseline w `results/binding-baseline`. Szczegóły różnicy różnic, neutralnego warunku `opisany` i transferu przy stałym pytaniu: [BINDING_PROTOCOL.md](BINDING_PROTOCOL.md).

`analysis.json` jest opisową agregacją zapisanych kontrastów i `transfers.csv`, wykonaną po pomiarze. Nie wprowadza nowych inferencji modelu. Weryfikator niezależnie przelicza równania interakcji i transferu z CSV.


## Pierwotny pomiar korpusowy

```powershell
python bernoulli_layers.py --model models/gollem-v6 --backend gollem --dataset PiotrSty/wolne-lektury-polish-literature-corpus --max-texts 48 --max-windows 32 --max-len 128 --out runs/bernoulli --perm 3000
```

Zachowano CSV i PNG tego etapu. Nie zapisano rewizji korpusu ani pełnego wyboru tekstów/okien, więc ponowne pobranie aktualnej wersji danych może dać inny wynik. Liczby z permutacji po warstwach traktujemy jako dawną diagnostykę, nie skalibrowane prawdopodobieństwa. Ograniczenia końcowego stanu i wykrywania backendu HF opisano w `METHODOLOGY.md`.

## Kontrola zachowanych artefaktów bez modelu

```powershell
python verify_results.py
```

To narzędzie standardowej biblioteki Pythona: sprawdza inwentarz SHA-256, odtwarza liczniki z JSON/CSV, sprawdza normy faktycznie zapisanych interwencji oraz brak wpływu identycznych stanów faktów w eksperymencie zaworu. Nie powtarza inferencji, nie ocenia jakości statystycznej eksperymentu i nie odzyskuje usuniętych danych.

Oryginalne metadane zawierają lokalne ścieżki do cache modelu. Są zachowane jako zapis pochodzenia; nie trzeba odtwarzać ich na swoim komputerze. Rewizja modelu jest podana niezależnie.

## Zakres archiwum

Zachowano cztery skrypty badawcze, wszystkie dostępne pełne wyniki, dostępne przebiegi smoke i baseline, pilotaże, źródła i pierwotną notatkę. Nie zachowały się surowe artefakty pierwotnego Bernoulli smoke, testu GPT-2 ani usunięte tymczasowe harnessy. Ich znane obserwacje zapisano w dzienniku, bez fabrykowania brakujących plików. Archiwum nie zawiera wag, środowiska wirtualnego, surowego korpusu ani pełnego eksportu rozmowy.
