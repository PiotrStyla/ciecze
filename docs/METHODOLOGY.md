# Metodologia i ograniczenia

## 1. Co oznacza analogia z płynem

Warstwy można modelować jako dyskretne kroki dynamiki, a reprezentacje tokenów jako oddziałujące cząstki. To nie utożsamia automatycznie transformera z nieściśliwą cieczą. Attention może łączyć odległe pozycje, a projekcje wartości, MLP i połączenia rezydualne zmieniają reprezentacje bez fizycznej zasady zachowania masy lub energii.

W obrazie aktywnego mikromiksera attention odpowiada wymianie między pozycjami, MLP lokalnemu przekształcaniu już dostępnego stanu, a residual stream przenosi wcześniejszy stan wraz z aktualizacjami. To użyteczna analogia organizacyjna, nie nowe równanie fizyczne.

## 2. Pierwotny test Bernoulliego

W `bernoulli_layers.py`:

- `v`: średnia norma przyrostu reprezentacji między stanami;
- `v_rel`: średni iloraz normy przyrostu i normy stanu;
- `A`: participation ratio widma kowariancji tokenów, liczone per okno i uśredniane;
- `p`: next-token cross-entropy z logit lens;
- `norm`: średnia norma reprezentacji.

PR jest liczony w float64, a jego sufit zależy od liczby obserwacji (po centrowaniu rząd jest nie większy niż liczba tokenów minus jeden). Nie jest fizycznym polem przekroju. CE nie ma wyprowadzonego związku z ciśnieniem.

CV iloczynu `v*A`, regresja `log v ~ log A` i dopasowanie `v²/2 = C-kp` są jedynie diagnostyką tych wyborów. Niespełnienie relacji nie obala wszystkich analogii płynowych. Ekstremalny pierwszy przyrost względny (~3869,8 w przebiegu korpusowym) może dominować dopasowanie. Swobodne permutacje zależnych, kolejnych warstw nie gwarantują skalibrowanego p-value; odjęcie liniowego trendu nie rozwiązuje całej zależności.

Dodatkowe ograniczenie ścieżki HF: skrypt traktuje ostatni `hidden_state` jako już znormalizowany, wcześniejsze jako surowe. Końcowy przyrost może zatem mieszać efekt bloku z końcową normalizacją. Automatyczne wykrywanie backendu opiera się na polach `n_embd/n_layer`; przy GPT-2 trzeba jawnie podać `--backend hf`. Dla GoLLeM pomiary `dye_flow.py` i `valve_flow.py` przechwytują stany bloków przed końcową normą i nie mają tej asymetrii.

## 3. Dwa barwniki

Każdy z sześciu układów zawiera trzy prompty:

1. czerwony klucz, zielony kubek;
2. niebieski klucz, zielony kubek;
3. czerwony klucz, niebieski kubek.

Pytanie dotyczy klucza. Osobno zmieniana jest kolejność faktów i liczba zdań pomiędzy nimi (0, 2, 6). Pary mają wyrównane pozycje tokenów. Szablon wybrano po pilocie na tym samym modelu; wyniki pilota, także nieudane, zachowano w `results/dye/pilot.json`.

Ślad to norma różnicy aktywacji podzielona przez średnią norm obu porównywanych stanów. To **wrażliwość**, a nie stężenie zachowanej substancji ani ilość informacji.

## 4. Zawór pytania

Każda para ma identyczny opis obu przedmiotów i identyczny prefiks odpowiedzi. Zmienia się tylko nazwa przedmiotu w pytaniu. `kluczyk` i `kubek` mają po dwa tokeny; pierwotna para `klucz/kubek` miała różną długość. Zachowano ten początkowy pilot w `results/valve/pilot.json`.

Pełny plan: 3 długości wstawki × 2 kolejności faktów × 2 przypisania kolorów = 12 par, 24 pytania. Nie odrzucano przypadków błędnie rozwiązywanych. `r` w identyfikatorze oznacza czerwony kluczyk, `b` niebieski.

Model jest przyczynowy. Późniejsze pytanie nie może zmienić stanów wcześniejszego opisu. Badamy więc odczyt i dalsze przekształcenia od miejsca pytania, a nie wsteczne przekierowanie całego przepływu. Niezerowy ślad w pytaniu nie jest dowodem poprawnego pobrania faktu.

## 5. Wynik odpowiedzi i interwencje

Oba eksperymenty używają kontrastu `log P(" czerwony") - log P(" niebieski")`, sumowanego po pełnych dwutokenowych odpowiedziach z teacher forcing. Dodatni znak oznacza preferencję czerwonego, ujemny niebieskiego. Nie jest to prawdopodobieństwo poprawności, swobodne generowanie ani pomiar względem całego słownika.

Stan 0 to wejście pierwszego bloku. Stan `l+1` to wyjście bloku `l`, przed `lnf`. Indeksy bloków w CSV są od zera. Patching wykonano w blokach 0, 5, 10, 15, 19, w obu kierunkach między promptami. Ślady obejmują wszystkie 21 stanów.

Podmieniany jest residual stream wybranych pozycji. Boczny GoLLeM `value_residual v0` pozostaje z przebiegu biorcy. Wynik opisuje tę konkretną interwencję w hybrydowy stan, nie pełną zmianę wszystkich pośrednich kanałów.

Kontrola null permutuje współrzędne różnicy między dawcą a biorcą i zachowuje normę perturbacji per pozycja. To jedna realizacja na interwencję, nie rozkład zerowy i nie p-value. Zaburzenia kontrolne nie muszą pozostawać na rozmaitości naturalnych aktywacji.

`toward_donor` oznacza przesunięcie w stronę **faktycznego wyniku** dawcy, nawet gdy jest błędny. `gap_fraction` nie jest prawdopodobieństwem i może przekraczać 1. Dla bardzo małej różnicy wyników nie jest raportowane. Duży efekt podmiany końcowego stanu odczytu jest oczekiwany konstrukcyjnie; sam nie odkrywa kanału przyczynowego. Podmiana innej pozycji po ostatnim bloku nie ma już kolejnego attention do przeniesienia zmiany.

## 6. Kontrole i zakres wnioskowania

W czasie rzeczywistych przebiegów sprawdzono:

- wyrównanie tokenów i brak zmiany wcześniejszych pozycji przy zmianie późniejszego tekstu;
- zgodność prefiksów przy teacher forcing dwóch odpowiedzi;
- niezmienność wyniku po podmianie aktywacji na nią samą;
- zgodność norm rzeczywistych perturbacji null/dawcy;
- w eksperymencie zaworu: zerową zmianę wyniku przy podmianie identycznych stanów faktów.

Interwencje są dowodem wpływu w danym obliczeniu, nie dowodem jedynej ścieżki, konieczności lub uogólnienia. Jeden checkpoint, mała rodzina syntetycznych promptów i dobór szablonu po pilocie nie pozwalają ekstrapolować wyników na wszystkie LLM. Nie wyznaczono liczby Reynoldsa, lepkości ani fizycznej turbulencji.
