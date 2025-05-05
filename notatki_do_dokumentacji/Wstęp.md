
![[Pasted image 20250426121415.png]]

### 1. Przygotowanie danych
- bierzemy partie szachowe z np. chesscom w formie PGN 
- konwertujemy dane na format [[FEN]] - robi to za nas Python - Chess (warto doczytać co ta klasa jeszcze robi)
- Z zapisów partii wyciągamy wszystkie możliwe pozycje pionków
- Duplikaty są usuwane
- Używamy [[Stockfish]] jako doradcy. 
- 
### 2. Zbiór danych 

- Zbiór treningowy - nie wiem ile chcemy
- Zbiór testowy - użyli 1k, by sprawdzić czy się nauczył. ok. 14% pokrywa się ze zbiorem treningowym
- Puzzle testowe 
	- 10k zadań , gdzie znane są prawidłowe ruchy
	- 1,33% nakłada się ze zbiorem treningowym


### 3. Jak się uczą modele?
 - Wartość pozycji (`State-value`): szansa na wygraną z tej pozycji. Wybieramy ruch, który **prowadzi** do pozycji o najlepszej wartości. -
- Wartość ruchów (`Action-values`): jak dobre są wszystkie legalne ruchy z tej pozycji. Wybieramy ruch, który **ma** największą wartość.
- Najlepszy ruch (`Best Action`): ruch, który **Stockfish** uznaje za najlepszy. Wybieramy ruch, który ma największe prawdopodobieństwo według modelu.

