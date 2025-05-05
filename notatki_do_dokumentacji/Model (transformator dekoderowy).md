 q
- dostaje sekwencję - szachowe poprzednie ruchy
- generuje nową sekwencję - kolejny ruch
Sekwencja w szachach - zmiana ruchu z 1 do 2 np. `e4 na e5`



Posiadamy 3 kluczowe warstwy dekodowania:
1. Self-Attention (Samo-Uwagę)
	- Model analizuje zależności między ***wszystkimi*** elementami w sekwencji, czyli nie patrzy na każdy element osobno, tylko jak ze sobą współgrają - **nadaje wagi parametrom!**
	- *Przykład* w szachach:
	    - Model widzi, że ruch pionka z `e2` na `e4` otwiera linię dla gońca na `f1`, ale też naraża króla na ewentualny szach z `h4`.
		- Rozumie, że jeśli przeciwnik zagrał `Sf6`, to Twój pion na `e5` jest atakowany.
	- W praktyce: Oblicza **ważność każdego elementu względem innych** (np. "król jest ważniejszy niż pion w końcówce"). Zadaje sobie pytania tj. 
		- Czy to otwiera gońca?
		- Czy król jest bezpieczny?
		- Czy pionki się wspierają?
	- **Self-attention** = które figury wpływają na siebie najbardziej
	
2. Feed-Forward Neural Network (FFN)
	- każdy element (token) przechodzi przez FFN
	- Po "zrozumieniu" relacji między elementami, każdy token jest przetwarzany przez **prostą sieć neuronową**, która dodaje **nieliniowe przekształcenia** (nie tylko analiza, ale także ocena konsekwencji i ukrytych motywów). Czyli można powiedzieć, że zaczyna sam myśleć np. czy coś jest opłacalne lub czy przeciwnik zauważy moją słabość

3. Residual/Skip Connections & Layer Normalization**
	Pomaga w stabilnym uczeniu – zapobiega [[zanikaniu gradientów]]. Po prostu dostaje amnezji i zapomina co miał w danych ruchu zrobić.
	- ==Residual/Skip Connections== - pomaga w uczeniu bardzo głębokich modeli. Uczymy model pewnego skrótu myślowego. Model uczy się różnicy - ***residuum*** (numerki!) między wejściem a wyjściem, a nie całej transformacji.
		- zamiast : `wyjście = Warstwa(wejście)`
		- mamy : `wyjście = Warstwa(wejście) + wejście` ,
			- `Warstwa(wejście)` to podstawowa operacja w sieci neuronowej, która przekształca dane wejściowe (np. pozycję szachową w formie FEN) w nową reprezentację, używając:
				-  **Wag (parametrów)** – których model się uczy - layer 1
				-  **Funkcji aktywacji** – która wprowadza nieliniowość  - layer 2
		- *Przykład:*
			- Wejście (`x`): "Król jest na g1, wieża na a1". 
			- Nadajemy wagi pionkom
			- `Warstwa(x)` wnioskowanie: "Król jest niechroniony przed atakiem z `h4`" - FFN
			- `Warstwa(x) + x` zachowuje **zarówno** oryginalną informację, jak i nową analizę.
		W DUŻYM SKRÓCIE -> to skrót, ułatwia se życia 

	- ==Normalizacja== - wyrównujemy dane, żeby trening był stabilniejszy.
		- *Przykład:*
			- Wejście: Wartości aktywacji z warstwy (np. wyniki self-attention w transformatorze).
			- Oblicz średnią i odchylenie standardowe dla tych wartości.
			- Normalizuj: Odejmij średnią, podziel przez odchylenie.
			- Przeskaluj: Pomnóż przez `γ` i dodaj `β` (by model mógł zachować ważne informacje)
	


### Sekwencja w transformerze - w skrócie
- Wejście: tokenizowany FEN (np. `rnbqkbnr/pppppppp/8/8...`)
- Analiza kontekstu - Model sprawdza zależności między warstwami sekwencji
	- Pozycja "e4" może być dobra, jeśli wcześniej wykonano "Sf3", ale zła po "f3". - czyli przechodzi przez warstwy
- Wyjście: generowanie/klasyfikacja – model przewiduje kolejne elementy (np. następny ruch w szachach) lub ocenia całą sekwencję (np. czy pozycja jest wygrana).


### Jak transformator radzi sobie z sekwencjami

- Jednocześnie analizuje wszystkie elementy sekwencji (np. wszystkie ruchy w partii szachowej), że nie tylko 1 ruch , a pare ruchów na raz - no po prostu myśli . Oblicza powiązania między pionkami - np.  jak ruszy `e4` to król będzie nadal bezpieczny
- Pamięta kolejność mimo, że przetwarza równolegle sekwencje, np. 
	- Bez kodowania: `e4` i `Sf3` byłyby traktowane jak "zbiór ruchów" bez kolejności.
	- Z kodowaniem: model wie, że `e4` było **przed** `Sf3`


### Czym się różni od pełnego transformatora?

Pełny składa się z 2 części 
- enkoder - analiza wejścia 
- dekoder - generuje wyjście 
Stockfish już nam nasze wejście przeanalizował 


