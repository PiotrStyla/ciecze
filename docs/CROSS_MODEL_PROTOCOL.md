# Porównanie między modelami: reguły ustalone przed pomiarem

Ten etap sprawdza, czy wynik zależy od jednego checkpointu. Prompt, siatka konfiguracji i reguły decyzyjne są ustalone **przed** policzeniem wyników dla drugiego modelu.

## Wynik

| Miara | SFT | Bazowy `SlayerLab/GoLLeM-v6-250M` |
|---|---:|---:|
| Ważne konfiguracje | 48 | 48 |
| Dodatnia interakcja `I` | 46/48 | 46/48 |
| Oba kierunki przeciwne | 19/48 | 9/48 |
| Średnie \|`I`\| | 0,5127 | 0,4089 |
| Średnie \|`C`\| | 0,6435 | 0,9271 |
| Kalibracja: dawca poza wszystkimi 20 losowaniami | 0/24 | 0/24 |
| Kalibracja: średnie odchylenie null | 0,2525 | 0,2834 |

Zgodny znak interakcji w 46 z 48 wspólnych konfiguracji, korelacja `I` r = 0,979, identyczny zbiór 16 odrzuceń mechanicznych. Rewizja modelu bazowego: `77d75a2b48eaab7aec59816f9849cee75a7bdfc1`.

- **Reguła 1 spełniona:** znak interakcji replikuje się w ≥75% konfiguracji w obu modelach.
- **Reguła 3 niespełniona w obu modelach:** 0/24 komórek, próg to 50%.
- Zgodnie z regułą 4 selektywny transfer przyczynowy zapisujemy jako **niepotwierdzony**.

## Modele

| Rola | Identyfikator | Pochodzenie |
|---|---|---|
| Po SFT | `PiotrSty/gollem-v6-250m-sft-answers`, rewizja `3a60edc280d93fb73ead0f5a5ba220986525aa76` | checkpoint wszystkich poprzednich etapów |
| Bazowy | `SlayerLab/GoLLeM-v6-250M` | model pretrenowany, ten sam kod i tokenizer; brak etapu SFT |

Ten sam `modeling_gollem_v6.py` i ten sam tokenizer pozwalają użyć identycznych promptów bez ponownej walidacji tokenizacji. Walidacja i tak jest wykonywana mechanicznie przy każdym przebiegu.

Rozwiązana rewizja modelu bazowego jest zapisywana w `summary.json` jako `resolved_snapshot` (ścieżka snapshotu zawiera SHA commita). Nie przypinamy jej z góry, bo etap jest nowy; dla modelu z poprzednich etapów przypięcie zostaje.

## Siatka i miary bez zmian

Ta sama siatka 4 szablony × 4 pary rzeczowników × 2 kolejności × 2 wstawki, ta sama walidacja tokenowa, ta sama miara `I = (R_key − R_cup)/2` i to samo znaczenie `C`.

Kalibracja: 20 losowań null na komórkę (dwa razy więcej niż poprzednio), bloki 5 i 10 — środek głębokości, gdzie wcześniej efekt był największy — na sześciu pierwotnych przypadkach.

## Reguły decyzyjne

Ustalone teraz, żeby nie dobierać kryterium po zobaczeniu liczb.

1. **Replikacja znaku interakcji** uznana, jeśli w obu modelach co najmniej 75% ważnych konfiguracji ma `I > 0`.
2. **Pełny wzorzec kierunków** (`R_key > 0` i `R_cup < 0`) raportowany jako proporcja; nie stawiamy progu, bo poprzedni etap dał 40%.
3. **Selektywny transfer przyczynowy** uznany za wspierany tylko wtedy, gdy dawca przekracza **wszystkie 20** losowań null w co najmniej **50%** komórek, w **obu** modelach. Przy 20 losowaniach najmniejsze osiągalne `p` wynosi 1/21 ≈ 0,048.
4. Jeśli warunek 3 nie jest spełniony w którymkolwiek modelu, wynik zapisujemy jako **niepotwierdzony**, a nie jako „bliski istotności”.
5. Żadnej korekty na wielokrotne porównania nie stosujemy, bo nie ogłaszamy pojedynczych istotnych komórek. Proporcje są miarą opisową.

## Ograniczenia

- Dwa checkpointy tej samej rodziny i tego samego tokenizera; nie jest to test międzyarchitektoniczny.
- Nadal jeden język, jedna rodzina zadań kolor–przedmiot i kontrast dwóch wymuszonych odpowiedzi.
- Model bazowy może po prostu nie umieć odpowiadać; wtedy jego wyniki opisują brak umiejętności, a nie mechanizm wiązania.
- 20 losowań to wciąż mało na mocne wnioski o małych efektach.
