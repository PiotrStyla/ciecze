# LLM jako aktywny mikromikser: źródła i propozycja

## Sprawdzone źródła pierwotne
- Lu et al., Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View, https://arxiv.org/abs/1906.02762. Interpretacja numerycznego solvera ODE konwekcji-dyfuzji, Lie-Trotter versus Strang-Marchuk, Macaron FFN-attention-FFN. Nie jest dowodem zachowania masy/energii w dowolnym LLM.
- Geshkovski et al., A mathematical perspective on Transformers, https://arxiv.org/html/2312.10794v2. Model cząstek, równanie ciągłości (3.4), uproszczone self-attention z normalizacją, często Q=K=V=I. Główne wyniki nie przenoszą się automatycznie na pełne causal multihead pre-norm LLM. Energia interakcji (3.5) rośnie dla atrakcyjnego modelu, nie jest energią Bernoulliego.
- Abnar & Zuidema, Quantifying Attention Flow in Transformers, https://aclanthology.org/2020.acl-main.385/. Attention rollout i attention flow są przybliżeniami pochodzenia wpływu; same wagi attention nie są rozstrzygającym wyjaśnieniem.
- Meng et al., Locating and Editing Factual Associations in GPT, https://rome.baulab.info/. Causal tracing: zakłócanie przebiegu i przywracanie aktywacji dla testowania wpływu na predykcję; to wcześniejsza metoda, nie nowość proponowanego eksperymentu.

## Korekta poprzedniej interpretacji
CE nie ma wyprowadzonej roli ciśnienia; participation ratio nie jest fizycznym polem przekroju. Nieudany fit nie falsyfikuje całej analogii płynowej. W raportowanych danych v_rel warstwy 0 ~3869.8 versus ~0.185–1.141 w innych warstwach; ekstremalny punkt uniemożliwia interpretację wysokiego R2 jako solidnej zależności rozłożonej po głębokości. Dowolne permutacje kolejnych, zależnych warstw nie dają automatycznie skalibrowanego testu istotności. W ścieżce HF ostatni hidden state jest traktowany jako po końcowej normalizacji, a wcześniejsze jako surowe, więc końcowy przyrost miesza normalizację ze zmianą residual stream.

## Propozycja (hipoteza, nie wynik)
Aktywny mikromikser z dwoma znacznikami: fakt istotny versus dystraktor. Minimalne pary promptów z identycznymi pozycjami tokenów, zmieniany jeden fakt, pytanie na końcu. Osobno manipulować pozycją faktu i liczbą dystraktorów w granicach kontekstu. Najpierw udowodnić, że model rozwiązuje zadanie bazowe.

Porównywać stany w tym samym punkcie bloków między dwoma przebiegami. Znormalizowana norma różnicy pokazuje wrażliwość, nie stężenie zachowanej substancji ani ilość informacji. Miara wyjściowa: różnica log-prawdopodobieństw dwóch odpowiedzi (pełnych sekwencji, jeśli wielotokenowe). Potwierdzać ślady przez patching aktywacji między przebiegami; efekt w hybrydzie nie dowodzi unikalnej ścieżki ani konieczności. Zamiana w obu kierunkach, kontrolne patche z innych przykładów, zmiany nieistotne o podobnej normie. Nie utożsamiać dużego attention z przyczynowym wpływem.

Hipoteza: dobre zachowanie wynika z selektywnego transportu istotnego sygnału, nie maksymalnego mieszania. Porażki można badać jako rozpraszanie, interferencję dystraktora albo brak wykorzystania dostępnego śladu. Nie ogłaszać turbulencji/Reynoldsa/Pécleta bez zdefiniowania wielkości i modelu fizycznego. Nowość zestawienia niepotwierdzona; jego elementy mają literaturę.

Nie zmieniano skryptu ani nie uruchamiano nowego eksperymentu w tym badaniu.
