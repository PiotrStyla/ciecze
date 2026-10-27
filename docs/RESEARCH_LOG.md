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
