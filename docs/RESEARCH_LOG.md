# Dziennik badań — 27 października 2026

To uporządkowany zapis pytań, decyzji, wyników i korekt z sesji badawczej, a nie dosłowny eksport czatu. Zawiera również wyniki negatywne. Surowe zachowane artefakty są w `results/`.

## 1. Intuicja: LLM, rura i woda

Punkt wyjścia: czy równanie Bernoulliego i efekt Venturiego opisują coś podobnego do działania warstw transformera? Nie ustalono bezpośredniego wyprowadzenia takich praw dla pełnego LLM. Znaleziono natomiast podstawy interpretacji warstw jako kroków dynamiki, tokenów jako cząstek i attention jako oddziaływania lub mieszania.

Początkowo zaproponowano umowne wskaźniki: norma przyrostu reprezentacji jako prędkość, participation ratio jako przekrój i CE logit lens jako ciśnienie. Powstał `bernoulli_layers.py`.

## 2. Urealnienie pierwszego skryptu

Początkowy identyfikator `PiotrSty/gollem-v5-128m` nie był potwierdzonym checkpointem do uruchomienia. Użyto rzeczywiście dostępnego `PiotrSty/gollem-v6-250m-sft-answers`.

Kod tego modelu jest natywnym modułem PyTorch, nie modelem HF AutoModel. Dodano backend GoLLeM i pozostawiono osobny backend HF. PR zaczęto liczyć per okno zamiast na wszystkich tokenach wszystkich dokumentów razem; usunięto przechowywanie całego korpusu aktywacji. Dodano limity liczby okien, float64 dla PR, wersję względnej prędkości oraz diagnostykę po odjęciu trendu głębokości.

Rzeczywiste sprawdzenia z sesji:

- PR macierzy rzędu 1: 1,0;
- logit lens ostatniego stanu GoLLeM i własne logity modelu dawały CE 3,6739, różnica 0;
- pierwszy przechwycony stan był zgodny z embeddingiem;
- backend HF uruchomiono z jawnym `--backend hf` na GPT-2.

Pomocnicze skrypty tych sprawdzeń oraz pliki pierwotnego GoLLeM smoke i GPT-2 zostały usunięte jeszcze podczas pracy, przed publikacją. Powyższe jest zapisem zaobserwowanego wyniku z sesji, nie odtworzonym surowym plikiem. Nie dorabiano brakujących CSV/PNG.

## 3. Bernoulli na korpusie

Polecenie wykonane w sesji:

```powershell
python bernoulli_layers.py --dataset PiotrSty/wolne-lektury-polish-literature-corpus --max-texts 48 --max-windows 32 --max-len 128 --out bernoulli_wolne --perm 3000
```

20 bloków, 32 okna, 4064 tokeny po pominięciu pierwszych tokenów okien. Zachowane wyniki: `results/bernoulli/layers.csv` i `layers.png`.

Zaobserwowane wskaźniki dla bezwzględnego przyrostu: CV(v*A) około 2,242, wobec CV(v) około 1,467 i CV(A) około 0,717. Nachylenie Venturiego około +0,91 zamiast postulowanego -1; po odjęciu liniowego trendu około +0,20. Dopasowanie Bernoulliego miało R² około 0,01. Dla względnego przyrostu duże R² miało przeciwny znak k i było zdominowane przez ekstremalny pierwszy punkt.

### Korekta wcześniejszego wniosku

Sformułowanie „hipoteza upada” było zbyt mocne. Wynik dotyczy konkretnych umownych wskaźników. CE nie zostało wyprowadzone jako ciśnienie, PR nie jest fizycznym przekrojem, a niepowodzenie takiego dopasowania nie obala wszelkich analogii z płynami. Permutacje zależnych warstw nie uzasadniają mocnych twierdzeń o istotności.

Nie zapisano rewizji korpusu, pełnego wyboru dokumentów/okien, sumy kodu w momencie tego przebiegu ani surowego logu. Aktualny skrypt jest zachowany, lecz nie deklarujemy bitowej odtwarzalności starego pomiaru.

## 4. Zmiana pomysłu: aktywny mikromikser

Zaproponowano dwa znaczniki: informację istotną i zakłócającą. Zamiast szukać zachowanej energii badamy rozchodzenie się wrażliwości i wpływ interwencji na odpowiedź. Pierwotną notatkę sprzed tych eksperymentów zachowano w `notes/fluid-theory.md` — końcowe zdanie o niewykonaniu eksperymentu odnosi się do momentu napisania tej notatki, nie do obecnego stanu repozytorium.

## 5. Dwa barwniki: klucz i kubek

Wybrany szablon: `Klucz jest [kolor]. ... Kubek jest [kolor]. Jaki jest klucz? Klucz jest`.

Pilot ujawnił silną zależność od sformułowania. Przykładowo wariant `Kolor klucza: ...` dawał odwrócone preferencje. Wszystkie pięć sprawdzonych szablonów z pilota zapisano w `results/dye/pilot.json`; nie jest to zbiór niezależny od doboru szablonu.

Pełny pomiar: sześć układów (dwie kolejności × 0, 2 lub 6 zdań), dwa rodzaje zmiany i interwencje w blokach 0, 5, 10, 15, 19. 7770 wierszy śladów i 960 interwencji łącznie z null.

| Kolejność | Zdań pomiędzy | Margines czerwony−niebieski: zielony kubek | Niebieski kubek |
|---|---:|---:|---:|
| klucz → kubek | 0 | 3,4252 | 1,6319 |
| klucz → kubek | 2 | 4,0066 | 1,3928 |
| klucz → kubek | 6 | 3,4933 | 0,7535 |
| kubek → klucz | 0 | 2,6401 | 0,9517 |
| kubek → klucz | 2 | 2,6020 | 1,3081 |
| kubek → klucz | 6 | 2,6095 | 1,2644 |

W obu wariantach koloru klucza poprawna odpowiedź była preferowana we wszystkich 6/6 układów. Zmiana nieistotnego koloru kubka osłabiała poprawną preferencję czerwonego klucza w każdym układzie. Podmiany aktywacji dystraktora również zmieniały wynik. Interpretacja robocza: interferencja przy zachowanej poprawnej preferencji, nie udowodnione całkowite „rozmycie” faktu.

Kontrole: maksymalny błąd self-patchingu około 9,54e-7, względny błąd norm null około 4,30e-7. Zachowano także wcześniejszy krótki przebieg w `results/smoke/dye` (128 interwencji). Ma on mniejszą rozdzielczość bloków i wcześniejszą pomocniczą kontrolę norm; głównym dowodem jest pełny przebieg.

## 6. Zawór: ten sam opis, inne pytanie

Początkowy pilot dla `klucz/kubek` ujawnił różną długość tokenizacji. Wyniki, również nieudane, są w `results/valve/pilot.json`. Do właściwego eksperymentu użyto `kluczyk/kubek`, oba po dwa tokeny w pytaniu. Nie dobierano dalej szablonu, żeby poprawić uzyskany wynik.

Plan: dwie kolejności faktów × trzy długości wstawki × dwa przypisania kolorów = 12 par, 24 pytania. Ten sam opis w parze, identyczny końcowy prefiks odpowiedzi. Zachowano wszystkie przypadki, nie tylko poprawne.

Wykonano baseline, krótki przebieg kontrolny, a następnie pełny pomiar: 8526 wierszy śladów i 960 interwencji.

| Pomiar | Wynik |
|---|---:|
| Poprawne preferencje pojedynczych pytań | 14/24 |
| Poprawne oba pytania w parze | 2/12 |
| Preferencja czerwonego | 20/24 |
| Kierunek zmiany po pytaniu zgodny z przypisaniem kolorów | 9/12 |
| Maksymalny błąd self-patchingu | 0 |
| Maksymalna względna różnica wcześniejszych stanów | 0 |
| Maksymalny względny błąd normy null | 1,96e-7 |
| Maksymalny wpływ podmiany identycznych stanów faktów | 0 |

Dla `key_first_f0_r` po bloku 10:

- pytanie o kubek: margines -0,6182 → +0,8337 po wstawieniu reprezentacji nazwy z pytania o kluczyk;
- pytanie o kluczyk: +0,3784 → -0,4820 po podmianie odwrotnej.

W tym przypadku wybór koloru można przełączyć samą interwencją w pytanie. Po zamianie kolorów między przedmiotami efekt nie zapewniał niezawodnej poprawnej selekcji. Pytanie ma wpływ przyczynowy, ale nie udowodniono ogólnego „zaworu”.

## 7. Stan wniosków

1. Nie potwierdzono prawa Bernoulliego ani Venturiego dla warstw LLM.
2. Transport i mieszanie mają podstawy w literaturze modeli dynamiki transformerów, z ważnymi założeniami.
3. Dostępność informacji w kontekście i poprawny jej odczyt to różne problemy.
4. Późniejsze pytanie nie może zmienić wcześniejszych stanów opisu w modelu przyczynowym. Analogią jest raczej selekcja na wyjściu niż ponowne mieszanie całej cieczy.
5. Do dalszego badania selektywnego transportu potrzebny jest model i zadanie o stabilnej jakości bazowej, dodatkowe szablony oraz powtarzane kontrole. Te dalsze badania nie zostały wykonane w tej sesji.

## 8. Odczyt faktu czy skojarzenie: kontrola treści

Do tej samej rodziny pytań dodano trzy warunki opisu: A = czerwony kluczyk i niebieski kubek, B = przypisanie odwrotne, N = oba przedmioty „opisany”, bez wskazania barwy. Słowo neutralne wybrano ze względu na brak podanej barwy i zgodną długość tokenizacji, nie na trafność odpowiedzi. Nie dobierano nowego szablonu.

Sześć ustawień kolejności/długości × trzy opisy × dwa pytania = 36 promptów. W pełnym przebiegu wykonano 720 interwencji, z których utworzono 360 sparowanych miar transferu. Osobno zachowano baseline bez interwencji i krótki smoke. Model i rewizja jak poprzednio.

Interakcja `I = ((m(A,key)-m(B,key))-(m(A,cup)-m(B,cup)))/2` usuwa addytywną stałą preferencję koloru i addytywny efekt samej nazwy pytania. Nie usuwa wszystkich heurystyk leksykalnych i pozycyjnych. Protokół: `BINDING_PROTOCOL.md`.

| Kolejność faktów | Zdań pomiędzy | R_key | R_cup | Interakcja I |
|---|---:|---:|---:|---:|
| kluczyk → kubek | 0 | 0,4006 | -0,4735 | 0,4370 |
| kubek → kluczyk | 0 | 0,1680 | -0,2008 | 0,1844 |
| kluczyk → kubek | 2 | 0,4233 | -0,9335 | 0,6784 |
| kubek → kluczyk | 2 | -0,5932 | -0,8190 | 0,1129 |
| kluczyk → kubek | 6 | 0,2548 | -1,5975 | 0,9261 |
| kubek → kluczyk | 6 | -0,1628 | -0,6153 | 0,2262 |

Interakcja jest dodatnia w 6/6 układów, średnio 0,4275. Silniejszy warunek przeciwnych kierunków (`R_key>0`, `R_cup<0`) zachodzi w 4/6. W dwóch pozostałych oba pytania reagują w tę samą stronę, ale różną siłą. W neutralnym opisie czerwony preferowany w 9/12 pytań; nie jest to test z prawidłową odpowiedzią ani bezwarunkowy prior.

Patching przenosi stany między opisami A/B przy niezmienionym pytaniu. Średnie selektywne przesunięcie po podmianie tokenów kolorów faktów wyniosło około 0,0577 po bloku 0, 0,0746 po bloku 5 i 0,0743 po bloku 10. Odpowiednie pojedyncze kontrole null miały średnie -0,1962, -0,0458 i -0,2309. To opisowe średnie, nie test istotności.

Podmiana nazwy w pytaniu między kontekstami przy niezmienionym jej brzmieniu dawała dużo mniejsze średnie selektywne efekty. Podmiana odczytu po bloku 15 dawała 0,2259 (null 0,0305), po bloku 19: 0,4266 (null 0,0294). Silny późny efekt odczytu jest konstrukcyjną kontrolą, nie odkryciem unikalnej ścieżki. Po ostatnim bloku podmiana innych pozycji nie ma już kolejnego attention do propagacji.

Kontrole: błąd self-patchingu 0, wcześniejsze stany przed pierwszym zmienionym kolorem bez różnicy, względny błąd norm null do 3,91e-7. Weryfikator przeliczył interakcje z surowych marginesów i selektywne transfery z par interwencji.

**Aktualizacja interpretacji:** niska trafność odpowiedzi nie oznacza całkowitego ignorowania przypisania przedmiot–kolor. Widać selektywny wpływ kontekstu, nakładający się na wspólną zmianę preferencji. Hipotezą roboczą jest rozróżnienie „sterowania wyborem strumienia” i „treści przenoszonego składnika”. Nie udowodniono semantycznego rozumienia, ogólnego mechanizmu zaworu ani praw mechaniki płynów w LLM.

## 9. Replikacja i kalibracja kontroli — korekta etapu czwartego

Etap czwarty opierał się na jednym szablonie, jednej parze przedmiotów i jednym losowaniu kontrolnym na interwencję. Etap piąty usuwa pierwsze dwie słabości replikacją, a trzecią powtarzanymi kontrolami.

### Replikacja zachowania bazowego

Zarejestrowano przed pomiarem 4 szablony × 4 pary rzeczowników męskich o równej tokenizacji × 2 kolejności × 2 długości wstawki = 64 konfiguracje. Walidacja mechaniczna odrzuciła wszystkie 16 konfiguracji szablonu T4 (`Kolorem jest X.`): wielka litera w `Niebieski` daje 3 tokeny zamiast 2, więc opisy A/B/N nie są wyrównane. Odrzucenia są zapisane z powodem, nie zastąpione innym szablonem.

Pozostało 48 konfiguracji:

| Miara | Wynik |
|---|---:|
| Dodatnia interakcja `I` | 46/48 |
| Oba pytania w przeciwnych, oczekiwanych kierunkach | 19/48 |
| T1 / T2 / T3: kierunki przeciwne | 9/16, 2/16, 8/16 |
| Średnie `I` | 0,5067 |
| Średnie \|`I`\| vs średnie \|`C`\| | 0,5127 vs 0,6435 |

T1 z parą `kluczyk/kubek` odtwarza etap czwarty co do czwartego miejsca (`I` = 0,4370 i 0,1844). To kontrola spójności implementacji.

**Wniosek:** sam znak interakcji replikuje się mocno, ale pełny wzorzec „oba pytania w przeciwnych kierunkach” tylko w 40% konfiguracji. Selektywna składowa jest też średnio mniejsza od wspólnej zmiany preferencji wywołanej zamianą kolorów.

### Kalibracja: jeden null to za mało

Dla konfiguracji T1 z parą `kluczyk/kubek` powtórzono pomiar z 10 niezależnymi, dopasowanymi normą zaburzeniami na każdą parę (kierunek, blok, pytanie). Kalibracja objęła wstawki 0 i 4 zdania oraz bloki 0, 5, 10 i 15 — 8 komórek na blok, łącznie 32 komórki. Nie była to pierwotna siatka 0/2/6; powtórzenie na niej i na modelu bazowym opisano w sekcji 10.

| Miara | Wynik |
|---|---:|
| Średni efekt selektywny dawcy | 0,0540 |
| Średnie odchylenie standardowe losowań null | 0,3017 |
| Dawca większy niż **wszystkie** 10 losowań | 2/32 |
| Losowania null o tym samym znaku co dawca | 209/320 |

**To unieważnia przyczynowy wniosek z etapu czwartego.** Mały transfer dodatni (+0,05 do +0,07) mieści się w rozrzucie losowych perturbacji tej samej normy. Poprzednie porównanie z jednym losowaniem trafiło na ujemne realizacje i sztucznie zawyżyło efekt.

Korekta: w etapie czwartym wolno twierdzić jedynie, że **zachowanie bazowe** zależy od przypisania kolorów do przedmiotu. Twierdzenie, że podmiana stanów kolorów przenosi tę zależność w sposób odróżnialny od losowego zaburzenia, **nie zostało potwierdzone**. Przy 10 losowaniach najmniejsze osiągalne `p` to 1/11 ≈ 0,09, więc nawet 2/32 nie jest mocnym dowodem; potrzeba więcej losowań i wielu niezależnych promptów.

### Co zostaje

1. Replikacja znaku interakcji na 3 szablonach i 4 parach jest solidna w granicach tego modelu.
2. Pełny wzorzec poprawnych kierunków nie replikuje się.
3. Selektywny transfer przyczynowy nie przeżył kalibracji null.
4. Wniosek metodologiczny jest tu ważniejszy od pozytywnego wyniku: pojedyncza kontrola losowa nie wystarcza do twierdzeń o patchingu.

## 10. Drugi model i kalibracja z 20 losowaniami

Reguły zapisano w `CROSS_MODEL_PROTOCOL.md` przed uruchomieniem. Dwa błędy wykryte w trakcie tego etapu:

- opcja `--null-fillers` istniała, ale nie była używana, więc pierwsza kalibracja objęła wstawki 0 i 4, a nie pierwotne 0/2/6. Poprawiono kod, opis poprawiono, a stary przebieg zachowano jako `results/null-calibration-10draws-f0f4` z jawną adnotacją;
- `--revision` miało domyślnie przypiętą rewizję modelu z badania, więc dla modelu bazowego trzeba było najpierw zezwolić na jego własną rewizję.

### Replikacja na modelu bazowym

`SlayerLab/GoLLeM-v6-250M`, rozwiązana rewizja `77d75a2b48eaab7aec59816f9849cee75a7bdfc1`, ten sam kod i tokenizer.

| Miara | SFT | Bazowy |
|---|---:|---:|
| Ważne konfiguracje | 48 | 48 |
| Dodatnia interakcja `I` | 46/48 | 46/48 |
| Oba kierunki przeciwne | 19/48 | 9/48 |
| Średnie \|`I`\| | 0,5127 | 0,4089 |
| Średnie \|`C`\| | 0,6435 | 0,9271 |

Te same 48 konfiguracji w obu modelach: zgodny znak w 46/48, korelacja `I` **r = 0,979**. Identyczny zbiór 16 odrzuceń mechanicznych.

**Wniosek:** wzorzec interakcji jest niemal identyczny przed i po SFT. To właściwość architektury, tokenizera i pretreningu, a nie efekt dostrajania. Reguła 1 (≥75% dodatnich w obu modelach) jest spełniona.

### Kalibracja z 20 losowaniami na obu modelach

Konfiguracja pierwotna (T1, `kluczyk/kubek`, wstawki 0/2/6), bloki 5 i 10, 24 komórki, 20 losowań null na komórkę.

| Miara | SFT | Bazowy |
|---|---:|---:|
| Średni efekt selektywny dawcy | 0,0745 | 0,0538 |
| Średnie odchylenie standardowe null | 0,2525 | 0,2834 |
| Dawca większy niż wszystkie 20 losowań | **0/24** | **0/24** |
| Losowania o znaku dawcy | 252/480 | 249/480 |
| Najmniejsze osiągalne `p` | 0,0476 | 0,0476 |

**Reguła 3 nie jest spełniona w żadnym modelu.** Efekt selektywny dawcy leży całkowicie wewnątrz rozrzutu losowych perturbacji o tej samej normie. Zgodnie z regułą zapisaną przed pomiarem wynik zapisujemy jako **niepotwierdzony**, a nie jako „bliski istotności”.

### Stan po tym etapie

1. Wzorzec zależności od przypisania przedmiot–kolor replikuje się między dwoma checkpointami z `r = 0,979` — to najsolidniejszy wynik ilościowy w repozytorium.
2. Pełny wzorzec poprawnych kierunków nadal słaby: 19/48 po SFT, 9/48 w modelu bazowym.
3. Przyczynowy transfer przez podmianę stanów kolorów **nie został potwierdzony** w żadnym z dwóch modeli.
4. Jedyna obrona tej hipotezy wymagałaby innej miary interwencji: mniejszej perturbacji, kontroli z parafrazą tej samej treści albo pomiaru względem normy różnicy. Obecna metoda jest zbyt szumiąca, by rozstrzygnąć efekt tej wielkości.

