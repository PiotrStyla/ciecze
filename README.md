# Ciecze × LLM

Eksploracyjne badanie analogii między przepływem płynów a przetwarzaniem informacji w transformerach. Od Bernoulliego i Venturiego do dwóch „barwników” i „zaworu” sterowanego pytaniem.

**To nie jest potwierdzona hydrodynamiczna teoria LLM.** Repozytorium zachowuje również wyniki negatywne, nieudane szablony i korekty zbyt mocnych interpretacji.

Badania wykonano 27 października 2026 na `PiotrSty/gollem-v6-250m-sft-answers`, CPU. Dokładna rewizja modelu: `3a60edc280d93fb73ead0f5a5ba220986525aa76`.

## Najważniejsze wyniki

| Etap | Co sprawdzono | Wynik i granica interpretacji |
|---|---|---|
| Bernoulli / Venturi | Przyrost reprezentacji, participation ratio, CE logit lens | Wybrane wskaźniki nie dały przekonującej analogii. CE nie jest wyprowadzonym ciśnieniem, PR nie jest fizycznym przekrojem. |
| Dwa barwniki | Zmiana faktu o kluczu kontra zmiana koloru kubka | Oba kolory klucza rozróżniane w 6/6 układów; niebieski kubek osłabiał preferencję czerwonego klucza we wszystkich sześciu. |
| Zawór pytania | Identyczny opis, pytanie o kluczyk albo kubek, zamiana kolorów i kolejności faktów | 14/24 poprawnych preferencji; oba pytania poprawne tylko w 2/12 układów. W 20/24 pytaniach preferowany czerwony. |

„Poprawna preferencja” oznacza większe prawdopodobieństwo właściwej spośród **dwóch wymuszonych pełnych odpowiedzi**. Nie jest to trafność swobodnego generowania. Próbki nie są niezależnym benchmarkiem.

W jednym poprawnie rozwiązywanym przypadku dwukierunkowa podmiana aktywacji nazwy przedmiotu w pytaniu po bloku 10 odwracała wybór koloru bez zmiany opisu. Efekt nie dawał niezawodnej poprawnej selekcji po zamianie kolorów między przedmiotami. Pytanie ma wpływ przyczynowy; nie dowiedziono uniwersalnego „zaworu”.

## Zawartość

```text
bernoulli_layers.py       # pierwotny pomiar wskaźników, backendy GoLLeM i HF
 dye_flow.py              # zmiana faktu / dystraktora, ślady i podmiany stanów
 valve_flow.py            # stały opis, zmienne pytanie, kontrbalansowanie
 verify_results.py        # sprawdzenie zapisanych wyników i sum SHA-256
 requirements.txt         # wersje głównych pakietów z użytego środowiska
 docs/                    # metodologia, przebieg, źródła, odtwarzanie
 notes/                   # pierwotna notatka koncepcyjna sprzed eksperymentów
 reproducibility/         # środowisko i inwentarz artefaktów z sumami SHA-256
 results/bernoulli/        # zachowane CSV i PNG z korpusu Wolnych Lektur
 results/dye/             # pełny przebieg dwóch barwników, w tym pilot
 results/valve/           # pełny przebieg zaworu, pilot, analiza i wykresy
 results/smoke/           # zachowane krótkie przebiegi kontrolne
```

Wagi modelu (~1,1 GB), cache Hugging Face, środowiska Python i surowy korpus książek nie są redystrybuowane. Trzy skrypty badawcze i istniejące artefakty skopiowano bez zmiany ich treści; sumy znajdują się w [inwentarzu](reproducibility/manifest.json).

## Oglądanie wyników

- [Przebieg i korekty interpretacji](docs/RESEARCH_LOG.md)
- [Metodologia i ograniczenia](docs/METHODOLOGY.md)
- [Literatura pierwotna](docs/SOURCES.md)
- [Dwa barwniki: przykład z sześcioma zdaniami](results/dye/source_first_f6.png)
- [Zawór: przegląd wszystkich układów](results/valve/overview.png)
- [Zawór: udany przypadek i dwukierunkowe interwencje](results/valve/key_first_f0_r.png)
- [Analiza ilościowa zaworu](results/valve/analysis.json)

## Uruchomienie

Python 3.11; dokładne obserwowane wersje: [environment.json](reproducibility/environment.json).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python verify_results.py
```

Domyślne uruchomienie skryptów pobiera aktualnie dostępną rewizję modelu z Hub. Aby użyć **rewizji z badania**, najpierw pobierz przypięty snapshot:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('PiotrSty/gollem-v6-250m-sft-answers', revision='3a60edc280d93fb73ead0f5a5ba220986525aa76', local_dir='models/gollem-v6', allow_patterns=['config.json','model.safetensors','tokenizer.json','modeling_gollem_v6.py'])"
python dye_flow.py --model models/gollem-v6 --out runs/dye
python valve_flow.py --model models/gollem-v6 --out runs/valve
```

Ładowanie GoLLeMa wykonuje kod Pythona z checkpointa. Sprawdź źródło przed użyciem obcego modelu. Szczegóły odtwarzania, polecenie dla korpusu i znane braki: [REPRODUCING.md](docs/REPRODUCING.md). Instalacja w nowym, pustym środowisku nie była osobno weryfikowana; zapisano działające środowisko oraz rzeczywiste przebiegi.
