# Zamrożony protokół: poprawny odczyt przed interwencją

Data planu: 28 października 2026. Ten dokument i `heldout_cases.json` powstały przed wykonaniem nowych pomiarów. Dotychczasowe wyniki GoLLeM są eksploracyjne; nowe pary rzeczowników nie pokrywają się z poprzednią siatką.

## Dobór modelu i bramka

1. Na 12 parach kalibracyjnych (`heldout_cases.json`, `gate`) mierzymy po dwa przypisania kolorów na parę i pytania o oba przedmioty: 24 opisy, 48 odpowiedzi na model. Najpierw GoLLeM po SFT (`PiotrSty/gollem-v6-250m-sft-answers` @ `3a60edc280d93fb73ead0f5a5ba220986525aa76`). Jeśli nie przejdzie, `Qwen/Qwen2.5-0.5B-Instruct`; jeśli nie przejdzie, `Qwen/Qwen2.5-1.5B-Instruct`. Rewizje modeli Qwen zapisujemy po rozwiązaniu przez Hub; nie zmieniamy promptów ani progu po wyniku.
2. Dla pytania liczymy różnicę log-prawdopodobieństw pełnych odpowiedzi `czerwony` i `niebieski` oraz osobno generujemy odpowiedź zachłannie. Poprawna odpowiedź swobodna musi zaczynać się dokładnie właściwym słowem (po odrzuceniu białych znaków i cudzysłowów; dopuszczalna kropka/przecinek na końcu). Inne słowo, odmiana, brak odpowiedzi i dodatkowe słowa to błędy. Wynik dotyczy wymuszonego prefiksu „Odpowiedz jednym słowem”, nie ogólnej konwersacji.
3. Opis zaliczony tylko jeśli **oba** pytania dają poprawną odpowiedź. Bramka: co najmniej **18/24 opisów poprawnych** jednocześnie według kontrastu odpowiedzi wymuszonych i odpowiedzi swobodnych (raportujemy liczniki osobno). Weryfikujemy tokenizację pełnych odpowiedzi; jeśli odpowiedź zmienia tokenizację prefiksu, konfiguracja jest błędna, a nie odrzucona po wyniku.
4. Model przechodzący bramkę jako pierwszy jest jedynym modelem w głównym eksperymencie. Jeśli żaden nie przejdzie, nie interpretujemy interwencji i raportujemy same wyniki bramki.

## Odłożone 20 par — pomiar po wyborze modelu

Dopiero po wyborze modelu otwieramy `heldout` z pliku. Każda z 20 par tworzy dwa przypisania czerwony/niebieski z jednakową liczbą obu kolorów i tymi samymi pytaniami o dwa przedmioty. Mierzymy osobno poprawność wymuszoną i swobodną. Parę (nie pytanie ani wariant A/B) traktujemy jako jednostkę niezależną w przedziałach niepewności. To nie jest benchmark języka polskiego: jeden schemat, wymuszony kolor i 40 opisów z 20 par.

## Interwencja, tylko jeśli bramka przeszła

Biorca to opis `X jest czerwony. Y jest niebieski.` (lub zamienione kolory). Dwa równoległe przebiegi dawcy mają **identyczny alternatywny szablon** `X ma kolor czerwony. Y ma kolor niebieski.`: (S) te same fakty, (O) kolory zamienione. Pytanie i jego cel są identyczne. W każdym z trzech opisów podmieniamy wyłącznie dwa stany wyjściowe wskazanego bloku na **tokenach kończących zdania faktów (kropkach)**, osobno dla obu pytań; w bocznych stanach attention pozostaje biorca. Nie podmieniamy pozycji pytania ani końcowego odczytu. Dopasowanie pozycji jest po offsetach znakowych odpowiadających końcom faktów, a nie po identycznym indeksie (parafraza ma inną liczbę tokenów). Wymagamy, by każda kropka mapowała się na dokładnie jeden token; sprawdzamy self-patch: ten sam stan musi dać tę samą predykcję z tolerancją 1e-4.

Blok 5: pierwotny. Blok 10: analiza dodatkowa bez roszczenia o osobną istotność. Dla każdego pytania `m = log P(poprawny kolor biorcy) − log P(drugi kolor)` po patchingu. Pierwotny efekt na parę to średnia po dwóch przypisaniach i obu pytaniach z `m(S) − m(O)` na bloku 5. Dodatnia wartość to efekt zgodny z różnicą treści między dawcami; nie dowodzi unikalnej ścieżki ani pełnego przeniesienia odpowiedzi. Raportujemy również m bez patchingu, normy `||S−biorca||` i `||O−biorca||` oraz surowe wyniki każdego przypadku; różnica norm ogranicza interpretację.

95% przedział bootstrap: 10000 losowań **20 par** ze zwracaniem, ziarno 20261028. Warunek pozytywnego wyniku: dolna granica przedziału > 0, co najmniej 15 z 20 par o efekcie dodatnim, i brak utraty poprawności bazowej na odłożonej próbie poniżej progu 30/40 opisów (oba pytania). Jeśli warunek zawiedzie, brak potwierdzenia przyczynowego wpływu treści w tej procedurze. Brak korekt na wielokrotne analizy, bo tylko blok 5 ma regułę decyzyjną.

Nie używamy wcześniejszych 48 układów, 24 komórek null ani nowych wyników odłożonych do doboru szablonów lub warstw. Zamiana kolorów nie rozwiązuje całkowicie wpływu ich pozycji i tokenizacji; wszystkie odrzucenia mechaniczne i ich powody raportujemy, bez cichego zastąpienia par.

**Poprawka mechaniczna przed pomiarem interwencji:** tokenizer Qwen dzieli `czerwony` i `niebieski` na różną liczbę tokenów (4 i 3 po spacji). Pierwotnie zapisane kryterium jednakowej długości spanu uniemożliwiałoby podmianę. Zmieniono pozycje interwencji na pojedyncze tokeny końca obu zdań faktów; zachowano te same przypadki, modeli, bloki, miary i regułę decyzyjną. Przed pełną bramką wykonano jeden dymny przebieg GoLLeM na pierwszej parze (nie służył do doboru progu). Rejestrujemy tę poprawkę jawnie, nie przedstawiamy jej jako ustalonej przed wszystkimi nowymi odczytami.

## Wynik i werdykt (po pomiarze)

| Model | Rewizja | Wymuszone: oba pytania | Swobodne: oba pytania | Bramka |
|---|---|---:|---:|---|
| GoLLeM SFT | `3a60edc280d93fb73ead0f5a5ba220986525aa76` | 6/24 | 0/24 | nie |
| Qwen 0,5B Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | 7/24 | 18/24 | nie |
| Qwen 1,5B Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 24/24 | 24/24 | **tak** |

Na odłożonych 20 parach Qwen 1,5B uzyskał 35/40 poprawnych **opisów** zarówno przy odpowiedziach wymuszonych, jak i swobodnych; 33/40 opisów poprawne jednocześnie w obu trybach. Pary z bramki nie należą do odłożonej próby. Wymuszony wynik Qwen 0,5B mógł być zniekształcony nierówną liczbą tokenów i wielkością liter w odpowiedzi: kontrast pełnych napisów jest miarą z góry wybraną, nie bezstronnym testem każdej strategii odpowiadania.

Na odłożonych parach wykonano 160 interwencji (20 par × 2 przypisania × 2 pytania × 2 bloki). Self-patch nie zmienił marginesu w żadnym przypadku (maksymalny błąd 0). **Pierwotna miara, blok 5:** średnie `m(S) − m(O)` po 20 niezależnych parach = **+0,0105** log-prawdopodobieństwa; przedział bootstrap 95% **[−0,0166; +0,0385]**, dodatnio w **10/20** parach. Wymagane było ≥15/20 oraz dolna granica >0; **reguła pozytywnego wyniku nie jest spełniona**. Blok 10 (dodatkowy): średnia −0,0161; bez odrębnego testu istotności.

W bloku 5 średnia norma podmiany wynosi 10,813 dla dawcy z tym samym faktem i 11,813 dla dawcy z odmiennym faktem; średnie bezwzględne zmiany marginesu względem braku patchingu to odpowiednio 0,3812 i 0,3880. Kontrola semantyczna nie jest więc dopasowana co do normy. **Nie wykazano selektywnego wpływu zmiany treści faktów przeniesionych przez stany końców zdań**, mimo że wybrany model dobrze odpowiadał na odłożonych promptach. To nie dowód, że informacja nie przepływa inną ścieżką. Wynik dotyczy jednego modelu, jednego szablonu zdań, dwóch bloków i dwóch tokenów końca faktów.

Surowe prompty, odpowiedzi swobodne, oba marginesy, rewizje, pozycje i normy: `results/heldout/*.json`. `verify_heldout.py` niezależnie przelicza liczniki, znaczenie dawców, efekty na parę, bootstrap i regułę decyzyjną.
