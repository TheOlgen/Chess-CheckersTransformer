
- ściągamy z 1000 mld gier np. Lichess , Chesscom, Kaggle 
- wyciągamy wszystkie możliwe pozycje 
- Każda pozycja oceniana Stockfishem. Obliczyli tzw. **"wartość pozycji"** (_state-value_), czyli szansę na zwycięstwo z tej pozycji - wyrażoną jako procent od 0% (pewna przegrana) do 100% (pewna wygrana). Obliczyli też **"wartości ruchów"** (_action-values_), czyli jak dobry jest każdy legalny ruch w tej pozycji. 
- Tryb Stockfisha max 50ms na ocenę pojedynczej pozycji , bez ograniczeń głębokości. Jak przekroczy limit czasowy, pozycja zostaje pominięta.
- Aby uniknąć sytuacji, że kilka pozycji z jednej gry trafi razem do treningu lub testu (co mogłoby zaburzyć uczciwość testów), **przemieszali losowo wszystkie pozycje po przygotowaniu danych**.
- to samo co wyżej opisane, tylko bardziej opisowo
##### Value binning
- vibe jak te przedziały na laborkach z MPwI
- Modele nie przewidują dokładnych procentów (np. 53,1782% szansy na wygraną).
- **Dzielą zakres od 0% do 100%** na **K klas** (domyślnie **K=128** klas).
- Każda klasa odpowiada pewnemu przedziałowi procentów, np.:
    - klasa 0: 0% – 0.78%    
    - klasa 1: 0.78% – 1.56%    
    - klasa 2: 1.56% – 2.34%   
    - itd.    
- Model przewiduje do której klasy należy wynik (a nie dokładną liczbę). Dzięki temu łatwiej trenować modele, bo zamieniają trudny problem regresji na łatwiejszy problem klasyfikacji.


### Tokenizacja
przekształcanie danych szachowych na formę zrozumiałą dla modelu 
https://www.kaggle.com/code/wlifferth/part-1-understanding-python-chess-and-fen




