# Replikacja i kalibracja kontroli

Piąty etap badań odpowiada na dwie słabości poprzednich wyników: wszystko opierało się na jednym szablonie i jednej parze przedmiotów, a każdą interwencję porównywano z jednym losowaniem kontrolnym.

## Replikacja: szablony × pary rzeczowników

Rejestr ustalono przed pomiarem: 4 szablony × 4 pary rzeczowników męskich × 2 kolejności faktów × 2 długości wstawki (0 i 4 zdania). Pary dobrano mechanicznie: oba słowa mają identyczną liczbę tokenów (`kluczyk/kubek` 2, `kot/pies` 1, `rower/samochód` 1, `ołówek/zeszyt` 3) i są rodzaju męskiego, więc przymiotniki `czerwony`/`niebieski` są gramatyczne w obu zdaniach.

Szablony:

| Id | Forma zdania faktu | Pytanie |
|---|---|---|
| T1 | `X jest kolorem.` | `Jaki jest X? Odpowiedź:` |
| T2 | `X jest kolorem.` | `Jaki kolor ma X? Odpowiedź:` |
| T3 | `X jest kolorem.` | `Pytanie: X? Odpowiedź:` |
| T4 | `Kolorem jest X.` | `Jaki jest X? Odpowiedź:` |

Każdy szablon odpytywany jest przy trzech opisach: **A** (pierwszy przedmiot czerwony, drugi niebieski), **B** (odwrotnie) i **N** (oba `opisany`, bez barwy). Miary `R_key`, `R_cup`, `I` i `C` są identyczne jak w etapie czwartym.

### Odrzucenia są wynikiem, nie sprzątaniem

Walidacja jest mechaniczna i wykonywana przed policzeniem marginesów:

1. oba pytania mają równą liczbę tokenów;
2. zmiana między pytaniami dotyczy wyłącznie nazwy przedmiotu;
3. opisy A, B i N mają identyczną długość tokenizacji;
4. zamiana A→B i A→N zmienia wyłącznie tokeny czterech przymiotników.

T4 (`Kolorem jest X.`) **nie przechodzi punktu 3**: wielka litera na początku zdania zmienia tokenizację formy żeńsko brzmiącej `Niebieski` — wielkie `Niebieski` ma 3 tokeny, małe `niebieski` 2. Konfiguracje T4 są raportowane jako odrzucone w `summary.json` i nie wchodzą do statystyk. Nie usuwamy ich po zobaczeniu wyników ani nie zamieniamy na inny szablon.

T1 z parą `kluczyk/kubek` odtwarza dokładnie wyniki etapu czwartego (`I` = 0,4370 i 0,1844 dla wstawki 0). To kontrola spójności implementacji, nie nowy dowód.

## Kalibracja null: rozkład zamiast jednego losowania

Dla pierwotnej konfiguracji (T1, `kluczyk/kubek`, wstawki 0/2/6, oba porządki) przenosimy aktywacje tokenów kolorów faktów między A i B przy niezmienionym pytaniu, w blokach 0/5/10/15. Dla każdej pary (kierunek, blok, pytanie) losujemy `R` niezależnych, dopasowanych normą zaburzeń i budujemy rozkład selektywnego przesunięcia `(d_key − d_cup)/2`.

Dla efektu dawcy liczymy dwustronne `p_emp = (1 + #{|null| ≥ |donor|}) / (R + 1)`.

Przy `R = 10` najmniejsze możliwe `p` wynosi `1/11 ≈ 0,09`. To **za mało**, by orzekać o małych efektach: test może jedynie pokazać, że efekt dawcy jest większy niż wszystkie losowania. Nie stosujemy korekty na wielokrotne porównania, bo raportujemy proporcje, a nie „istotne” pojedyncze komórki. Kontrole są losowane osobno dla każdego pytania, bo prompty różnią się treścią.

## Ograniczenia tego etapu

- jeden checkpoint, jedna rodzina zadań kolor–przedmiot;
- kontrast dwóch wymuszonych odpowiedzi, nie swobodne generowanie;
- rzeczowniki męskie i pary o równej tokenizacji — wynik nie mówi nic o innych częściach mowy;
- `opisany` jest gramatycznie dopuszczalny, ale semantycznie dziwny; to kontrola treści, nie bezwarunkowy prior;
- brak niezależnego zbioru testowego: szablon T1 i para `kluczyk/kubek` były już użyte w poprzednim etapie;
- dziesięć losowań null to za mało na mocne wnioski: najmniejsze osiągalne `p` wynosi 1/11 ≈ 0,09.

## Wynik, który zmienił interpretację

Kalibracja unieważniła przyczynowy wniosek z etapu czwartego. Średni efekt selektywny dawcy wyniósł 0,0540, a średnie odchylenie standardowe pojedynczych losowań null 0,3017. Dawca przekroczył **wszystkie** dziesięć losowań tylko w 2 z 32 komórek, a 209 z 320 losowań miało ten sam znak co dawca.

Wcześniejsze porównanie z jednym losowaniem na interwencję trafiało na ujemne realizacje i zawyżało efekt. Poprawna konkluzja etapu czwartego jest więc węższa: **zachowanie bazowe zależy od przypisania kolorów do przedmiotu, ale nie wykazano, że podmiana stanów przenosi tę zależność w sposób odróżnialny od losowego zaburzenia o tej samej normie.**

Zostaje wniosek metodologiczny: porównywanie patching z jednym losowaniem kontrolnym nie wystarcza do twierdzeń o wpływie przyczynowym.

## Uruchomienie

```powershell
python replicate_flow.py --part replicate --out runs/replicate
python replicate_flow.py --part nulls --null-draws 10 --out runs/nulls
```

Domyślnie model jest pobierany w przypiętej rewizji `3a60edc280d93fb73ead0f5a5ba220986525aa76`. Artefakty: `summary.json`, `cases.csv`, `traces.csv`, `nulls.csv`, `replication.png`, `nulls.png`.
