# Ciecze × LLM

Eksploracyjne badanie analogii między przepływem płynów a przetwarzaniem informacji w transformerach. Od Bernoulliego i Venturiego do dwóch „barwników” i „zaworu” sterowanego pytaniem.

**To nie jest potwierdzona hydrodynamiczna teoria LLM.** Repozytorium zachowuje również wyniki negatywne, nieudane szablony i korekty zbyt mocnych interpretacji.

Pierwsze etapy wykonano 27 października 2026 na `PiotrSty/gollem-v6-250m-sft-answers`, CPU (rewizja `3a60edc280d93fb73ead0f5a5ba220986525aa76`). Niezależny test odczytu z 28 października obejmuje też modele Qwen; wyników różnych architektur nie łączymy bezpośrednio.

## Najważniejsze wyniki

| Etap | Co sprawdzono | Wynik i granica interpretacji |
|---|---|---|
| Bernoulli / Venturi | Przyrost reprezentacji, participation ratio, CE logit lens | Wybrane wskaźniki nie dały przekonującej analogii. CE nie jest wyprowadzonym ciśnieniem, PR nie jest fizycznym przekrojem. |
| Dwa barwniki | Zmiana faktu o kluczu kontra zmiana koloru kubka | Oba kolory klucza rozróżniane w 6/6 układów; niebieski kubek osłabiał preferencję czerwonego klucza we wszystkich sześciu. |
| Zawór pytania | Identyczny opis, pytanie o kluczyk albo kubek, zamiana kolorów i kolejności faktów | 14/24 poprawnych preferencji; oba pytania poprawne tylko w 2/12 układów. W 20/24 pytaniach preferowany czerwony. |
| Odczyt czy skojarzenie | Stały zestaw kolorów, zamiana przypisania do przedmiotów, kontrola bez koloru | Dodatnia interakcja kontekstu i pytania w 6/6 układów; przeciwne kierunki odpowiedzi na zamianę w 4/6. To nie jest trafność ani dowód rozumienia. |
| Replikacja | 3 szablony × 4 pary rzeczowników × 2 kolejności × 2 wstawki, plus 10 losowań kontrolnych na komórkę | Znak interakcji replikuje się w 46/48 układów (T4 odrzucony mechanicznie). Pełny wzorzec poprawnych kierunków tylko 19/48. **Selektywny transfer przyczynowy nie przeżył kalibracji: 2/32 komórki poza wszystkimi losowaniami.** |
| Drugi model | Ta sama siatka i 20 losowań null na obu checkpointach, reguły zapisane przed pomiarem | Wzorzec interakcji niemal identyczny: 46/48 w obu, korelacja `I` **r = 0,979**. Kalibracja: **0/24** komórek poza losowaniami w obu modelach — transfer przyczynowy **niepotwierdzony**. |
| Odłożony test zadania | 12 nowych par do wyboru modelu; 20 innych par odłożonych do pomiaru | GoLLeM SFT: 6/24 poprawnych opisów według wymuszonej odpowiedzi, 0/24 swobodnie. Qwen 0,5B: 7/24 i 18/24. Qwen 1,5B: 24/24 i 24/24; na odłożonych parach 35/40 i 35/40. Dopiero dla tego modelu interpretujemy interwencję. |
| Kontrola z parafrazą na odłożonych parach | Qwen 1,5B, dawca z tym samym lub zamienionym faktem; podmiana stanów dwóch końców zdań | Efekt parowany po bloku 5: **+0,0105**, przedział bootstrap 95% **[−0,0166; +0,0385]**, dodatnio w **10/20** par. Z góry ustalona reguła nie przeszła: **nie potwierdzono** selektywnego wpływu przenoszonej treści tą metodą. |

W historycznych etapach „poprawna preferencja” oznacza większe prawdopodobieństwo właściwej spośród **dwóch wymuszonych pełnych odpowiedzi**. Nie jest to trafność swobodnego generowania. Nowy etap raportuje wymuszone i swobodne odpowiedzi **osobno**, ale nadal nie stanowi ogólnego benchmarku języka.

W jednym poprawnie rozwiązywanym przypadku dwukierunkowa podmiana aktywacji nazwy przedmiotu w pytaniu po bloku 10 odwracała wybór koloru bez zmiany opisu. Efekt nie dawał niezawodnej poprawnej selekcji po zamianie kolorów między przedmiotami. Pytanie ma wpływ przyczynowy; nie dowiedziono uniwersalnego „zaworu”.

## Zawartość

```text
bernoulli_layers.py       # pierwotny pomiar wskaźników, backendy GoLLeM i HF
 dye_flow.py              # zmiana faktu / dystraktora, ślady i podmiany stanów
 valve_flow.py            # stały opis, zmienne pytanie, kontrbalansowanie
 binding_flow.py          # kontrola treści opisu i selektywnego transferu faktów
 replicate_flow.py        # replikacja szablonów/par i kalibracja null
 heldout_flow.py         # niezależna bramka odpowiedzi i kontrolowane podmiany stanów
 heldout_cases.json       # niezależne pary gate/heldout, zdefiniowane przed pomiarem
 verify_heldout.py        # ponowne obliczenie bramki i wyniku z zapisanych odpowiedzi
 verify_results.py        # sprawdzenie zapisanych wyników i sum SHA-256
 requirements.txt         # wersje głównych pakietów z użytego środowiska
 docs/                    # metodologia, przebieg, źródła, odtwarzanie
 notes/                   # pierwotna notatka koncepcyjna sprzed eksperymentów
 reproducibility/         # środowisko i inwentarz artefaktów z sumami SHA-256
 results/bernoulli/        # zachowane CSV i PNG z korpusu Wolnych Lektur
 results/dye/             # pełny przebieg dwóch barwników, w tym pilot
 results/valve/           # pełny przebieg zaworu, pilot, analiza i wykresy
 results/binding/         # czwarty etap: 36 promptów, 720 interwencji
 results/binding-baseline/ # wcześniejsza kontrola bez interwencji
 results/replication/     # 48 układów replikacji, 16 odrzuceń mechanicznych
 results/base-replication/ # ta sama siatka na modelu bazowym (r = 0,979)
 results/null-calibration/# 20 losowań null na komórkę, model po SFT
 results/null-calibration-base/   # ta sama kalibracja, model bazowy
 results/null-calibration-10draws-f0f4/ # wcześniejsza kalibracja, wstawki 0/4
 results/heldout/          # 3 bramki modelowe i odłożona próba
 results/smoke/           # zachowane krótkie przebiegi kontrolne
```

Wagi modeli (GoLLeM ~1,1 GB, Qwen pobierany osobno), cache Hugging Face, środowiska Python i surowy korpus książek nie są redystrybuowane. Pierwotne trzy skrypty badawcze i ich artefakty zachowano bez zmiany treści; kolejne etapy dodają nowe skrypty i wyniki. Sumy wszystkich publikowanych plików znajdują się w [inwentarzu](reproducibility/manifest.json).

## Oglądanie wyników

- [Przebieg i korekty interpretacji](docs/RESEARCH_LOG.md)
- [Metodologia i ograniczenia](docs/METHODOLOGY.md)
- [Literatura pierwotna](docs/SOURCES.md)
- [Dwa barwniki: przykład z sześcioma zdaniami](results/dye/source_first_f6.png)
- [Zawór: przegląd wszystkich układów](results/valve/overview.png)
- [Zawór: udany przypadek i dwukierunkowe interwencje](results/valve/key_first_f0_r.png)
- [Analiza ilościowa zaworu](results/valve/analysis.json)
- [Protokół kontroli kontekstu i definicja interakcji](docs/BINDING_PROTOCOL.md)
- [Odczyt czy skojarzenie: wynik interwencji](results/binding/transfer.png)
- [Analiza ilościowa wiązania przedmiot–kolor](results/binding/analysis.json)
- [Protokół replikacji i kalibracji kontroli](docs/REPLICATION_PROTOCOL.md)
- [Reguły porównania między modelami i ich wynik](docs/CROSS_MODEL_PROTOCOL.md)
- [Niezależna bramka i porównanie dawcy o tym samym/odmiennym fakcie](docs/HELDOUT_PROTOCOL.md)
- [Surowe wyniki odłożonej próby i 160 interwencji](results/heldout/selected.json)
- [Replikacja: znak interakcji według szablonów i par](results/replication/replication.png)
- [Model bazowy: ta sama siatka](results/base-replication/summary.json)
- [Kalibracja: dawca kontra powtarzane losowania](results/null-calibration/nulls.png)
- [Podsumowanie kalibracji](results/null-calibration/analysis.json)

## Uruchomienie

Python 3.11; dokładne obserwowane wersje: [environment.json](reproducibility/environment.json).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python verify_results.py
```

Pierwotne trzy skrypty domyślnie pobierają aktualnie dostępną rewizję modelu z Hub. `binding_flow.py` domyślnie przypina rewizję z badania. Aby również pozostałe skrypty używały **tej samej rewizji**, najpierw pobierz snapshot:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('PiotrSty/gollem-v6-250m-sft-answers', revision='3a60edc280d93fb73ead0f5a5ba220986525aa76', local_dir='models/gollem-v6', allow_patterns=['config.json','model.safetensors','tokenizer.json','modeling_gollem_v6.py'])"
python dye_flow.py --model models/gollem-v6 --out runs/dye
python valve_flow.py --model models/gollem-v6 --out runs/valve
python binding_flow.py --model models/gollem-v6 --out runs/binding
```

Ładowanie GoLLeMa wykonuje kod Pythona z checkpointa. Sprawdź źródło przed użyciem obcego modelu. Szczegóły odtwarzania, polecenie dla korpusu i znane braki: [REPRODUCING.md](docs/REPRODUCING.md). Instalacja w nowym, pustym środowisku nie była osobno weryfikowana; zapisano działające środowisko oraz rzeczywiste przebiegi.
