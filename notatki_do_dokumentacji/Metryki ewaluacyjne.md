#### 1. Action-Accuracy (Dokładność wyboru akcji)

gdy ruch identyczny co Stockfish 
_Przykład_:
Jeśli Stockfish w 100 testowych pozycjach zalecał `e2e4`, a model wybrał ten ruch w 85 przypadkach, jego action-accuracy wynosi 85%.
Najlepszy model (270M) osiągnął 69.4%

#### 2. Action-Ranking (Kendall’s τ)

Korelacja między rankingiem ruchów modelu a rankingiem Stockfish
- Wartości: od `-1` (ranking odwrotny) do `1` (idealna zgodność).
- `0` oznacza brak korelacji.

 Dla każdej pozycji porównuje się, czy ruchy ułożone według:
- `Q̂(s,a)` (wartość akcji),
- `-V̂(s’)` (wartość stanu po ruchu),
- `P(a|s)` (klonowanie behawioralne)  
są zgodne z rankingiem Stockfish
Najlepszy model osiągnął τ = 0.300 (umiarkowana zgodność)


#### 3. Puzzle-Accuracy (Dokładność rozwiązywania zagadek)
Procent zagadek szachowych (np. z Lichess), gdzie **sekwencja ruchów modelu** dokładnie pokrywa się z rozwiązaniem.


#### 4. Lichess Elo (siła gry)
Siła modelu w skali Elo (im wyższe, tym lepiej).