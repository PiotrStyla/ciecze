# Odczyt faktu czy skojarzenie: protokół kontroli kontekstu

Czwarty etap badań. Pytanie: czy słaby wynik zadania zaworu oznacza zupełne ignorowanie przypisania kolorów, czy wpływ kontekstu jest obecny, lecz niewystarczający do odwrócenia końcowej preferencji odpowiedzi?

## Plan, bez dalszego doboru szablonu

Używamy tej samej rodziny promptów `kluczyk/kubek` co w poprzednim etapie. Nie szukamy nowego szablonu poprawiającego trafność. Kolejność faktów: dwie możliwości. Liczba zdań pomiędzy nimi: 0, 2, 6. Daje to sześć ustawień.

W każdym ustawieniu są trzy opisy i dwa pytania, łącznie sześć promptów:

- **A:** czerwony kluczyk, niebieski kubek;
- **B:** niebieski kluczyk, czerwony kubek;
- **N:** kluczyk opisany, kubek opisany — bez określenia koloru.

Każdy opis jest odpytywany o kluczyk i o kubek, z identycznym końcowym prefiksem `Odpowiedź:`. Słowa `czerwony`, `niebieski` i `opisany` mają w tych miejscach po dwa tokeny. Program wymaga, żeby przy zamianie A→B lub A→N zmieniały się wyłącznie cztery pozycje przymiotników, a nie pozycje pytania i odczytu.

Warunki A i B zawierają ten sam zestaw kolorów, więc sama ich liczebność nie rozróżnia opisów. Zmienia się przypisanie koloru do przedmiotu i położenie danego koloru. Kontrola N jest tekstem o innym znaczeniu, nie czystym bezwarunkowym priorem modelu. Nie przypisujemy jej „poprawnego” koloru.

## Miara interakcji

Niech `m` będzie różnicą log-prawdopodobieństw pełnych dwutokenowych odpowiedzi czerwony i niebieski, z teacher forcing.

```text
R_key = m(A, key) - m(B, key)
R_cup = m(A, cup) - m(B, cup)
I = (R_key - R_cup) / 2
C = (R_key + R_cup) / 2
```

`I` jest interakcją przypisania kolorów i celu pytania. Addytywna stała preferencja koloru i addytywny efekt samej nazwy pytania znoszą się w tej różnicy różnic. `C` opisuje wspólny efekt zmiany opisu, np. składową związaną z pozycją koloru.

Przy selektywnym wpływie w oczekiwanych kierunkach `R_key > 0` i `R_cup < 0`. Samo `I > 0` jest słabszym warunkiem: może zachodzić, gdy oba pytania przesuwają wynik w tę samą stronę, lecz o różną wielkość. Żaden z tych warunków nie wymaga przejścia marginesu przez zero. Dlatego dodatnia interakcja i niska trafność preferencji mogą współistnieć.

To nie jest miara ilości informacji, dowód rozumienia semantyki, p-value ani oszacowanie działania wszystkich LLM. Nie znosi dowolnych nieaddytywnych heurystyk leksykalnych i pozycyjnych.

## Interwencje: stałe pytanie, zmieniony dawca faktów

Patching zachodzi między A i B **przy niezmienionym pytaniu**. To inny test niż poprzednie przenoszenie nazwy przedmiotu między pytaniami.

Regiony:

1. obie pary tokenów określających kolory faktów razem;
2. tokeny nazwy przedmiotu w pytaniu;
3. końcowa pozycja odczytu.

Próbkowane bloki: 0, 5, 10, 15, 19, przed końcową normalizacją. Oba kierunki A→B i B→A. Dla każdej interwencji: prawdziwy dawca i jedno zaburzenie null przez permutację współrzędnych różnicy z zachowaniem normy. W bocznym `value_residual v0` pozostaje stan biorcy.

Dla każdego kierunku, regionu, bloku i rodzaju interwencji mierzymy parę przesunięć: `d_key` i `d_cup`. Pochodna miara transferu:

```text
selective_shift = sign * (d_key - d_cup) / 2
common_shift = sign * (d_key + d_cup) / 2
sign = +1 dla A→B, -1 dla B→A
```

Dodatni efekt selektywny jest zgodny z przenoszeniem wpływu przypisania kolorów. Nie musi być dodatni w każdym przypadku i nie jest procentem odzyskanej wiedzy. Średnie w wykresie są opisowe; nie mają przedziałów ufności ani deklarowanej istotności. Przypadki i oba kierunki są zachowane w CSV.

Kontrole w czasie inferencji: wyrównanie wszystkich pozycji, brak wpływu na tokeny przed pierwszym zmienionym kolorem, niezmienność po self-patchingu i zgodność norm zastosowanych perturbacji. Duży efekt późnego patchingu odczytu jest oczekiwany konstrukcyjnie. Brak późnego efektu patchingu kolorów lub nazwy nie dowodzi utraty informacji: może nie być już kolejnego attention, które przekaże lokalną zmianę do odczytu.

## Artefakty i uruchomienie

```powershell
python binding_flow.py --baseline-only --out runs/binding_baseline
python binding_flow.py --out runs/binding
```

Domyślny identyfikator modelu i rewizja są przypięte do checkpointa z dotychczasowych badań: `PiotrSty/gollem-v6-250m-sft-answers`, rewizja `3a60edc280d93fb73ead0f5a5ba220986525aa76`. Można podać już pobrany katalog przez `--model models/gollem-v6`.

`summary.json` przechowuje wszystkie prompty, tokeny, kontrasty i wyniki kontroli. `baselines.csv` zawiera pełne marginesy dla 36 promptów. `patches.csv` zawiera 720 interwencji pełnego planu, a `transfers.csv` 360 sparowanych kontrastów. `traces.csv` obejmuje porównania A–B i A–N dla obu pytań i wszystkich stanów. Tryb baseline-only nie wykonuje interwencji i zapisuje ich status jako null, nie zero.

Protokół i wzory ustalono przed odczytaniem pełnych wyników interwencji. Ten etap nadal korzysta z szablonu wybranego po dawnym pilocie i z wcześniejszych wyników zaworu, więc nie jest niezależną konfirmacją na nowym zbiorze.
