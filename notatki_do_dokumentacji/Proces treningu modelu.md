1. Przygotowanie danych 
	- na pewno na start otrzymujemy 6 mln rozgrywek w postaci FEN 
	- a potem nie wiem 
	- w artykule zapisywali w postaci UCI
	- chyba python-chess nam coś z tym robi ?
	- TODO
2. Architektura modelu 
	- transformator dekoderowy 
	- wejście:
		- Dla **wartości stanu/akcji**: Sekwencja 78 tokenów (FEN + dodatkowe informacje).
		- Dla **klonowania behawioralnego**: Sekwencja 77 tokenów (FEN).
	- wyjście:
		- Dla wartości: `K` przedziałów (binów) oceny (np. od -10 do +10).
		- Dla ruchów: 1968 możliwych akcji (softmax)
3. Funkcja straty
	- cel : minimalizacja **[[cross - entropy]]** (różnicy między przewidywaniami a rzeczywistością).  
4. Optymalizacja  ( [[Algorytm Adam]] ) 
