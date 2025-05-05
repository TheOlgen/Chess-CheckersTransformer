Model wykorzystuje trzy główne podejścia do podejmowania decyzji w szachach

#### 1. Predykcja wartości akcji (AV)

Ocenia jakość konkretnego ruchu 

wejście: tokenizowana pozycja 's' i konkretny ruch 'a'
wyjście: daje rozkład prawdopodobieństwa dla przedziałów wartości (MPwI lab 1 i 2 - przedziały )

 *Przykład*: dla ruchu e2e4 może przewidywać:
- 60% szans na wartość +1.0
- 30% szans na +1.5
- 10% szans na +0.5

Funkcja starty [[cross - entropy]]
$$-log P(z_i|s,a)$$
_z_i_ - przedział zawierający prawdziwą wartość ruchu
po polsku: "Logarytm prawdopodobieństwa, że wartość ruchu *a* w stanie *s* należy do przedziału _z_i_, pod warunkiem tokenizacji stanu i akcji"

Wybieramy ruch o najwyższej oczekiwanej wartości spośród wszystkich legalnych ruchów.

#### 2. Predykcja wartości stanu (SV)

Ocena ogólnej jakości pozycji (bez wskazywania konkretnego ruchu).

wejście: tylko tokenizowana pozycja _s_

 _Przykład_: dla pozycji początkowej może dać:
- 70% na wartość 0.0 (remis)    
- 20% na +0.5    
- 10% na -0.5

Dla każdego legalnego ruchu oblicz wynikową pozycję (s')
Wybierz ruch, który prowadzi do pozycji najgorszej dla przeciwnika (czyli najlepszej dla nas)



### 3. Klonowanie behawioralne (BC)
Naśladowanie ruchów ekspertów (z bazy danych)


wejście:  tokenizowana pozycja _s_
 _Przykład_: dla pozycji początkowej może dać:
	- e2e4: 45%  
	- d2d4: 30%
	- inne ruchy: łącznie 25%


Funkcja starty:
$$-log P(a|s)$$
Zawsze wybieramy ruch o najwyższym przewidywanym prawdopodobieństwie


| Cecha         | AV                         | SV                                  | BC                        |
| ------------- | -------------------------- | ----------------------------------- | ------------------------- |
| **Wejście**   | Pozycja + ruch             | Tylko pozycja                       | Tylko pozycja             |
| **Wyjście**   | Wartość ruchu (przedziały) | Wartość pozycji (przedziały)        | Prawdopodobieństwa ruchów |
| **Strategia** | Maksymalizuj wartość ruchu | Minimalizuj wartość dla przeciwnika | Naśladuj ekspertów        |
| **Użycie**    | Ocena konkretnych ruchów   | Ocena ogólnej sytuacji              | Uproszczona imitacja      |


_Przykłady_
 W pozycji początkowej:
    - AV: e2e4 ma wartość +0.7, d2d4 +0.5 → wybiera e2e4
    - SV: e2e4 prowadzi do pozycji o wartości -0.3 dla przeciwnika → wybiera e2e4
    - BC: e2e4 ma 45% prawdopodobieństwo, d2d4 30% → wybiera e2e4