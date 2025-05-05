

### Cross-Entropy dla Ruchów (Klasyfikacja)

Cel: Chcemy sprawić, by model przypisywał **jak najwyższe prawdopodobieństwo prawidłowym ruchom**.

bardziej szczegółowo:
- Model dla każdej pozycji szachowej oblicza **prawdopodobieństwa wszystkich ruchów** (np. `e2e4`: 70%, `d2d4`: 20%, ...).
- **Strata** mierzy, jak bardzo te przewidywania różnią się od rzeczywistości (np. czy `e2e4` miał najwyższe `p`).
- `Strata = -log(p)`
	- `p` = prawdopodobieństwo przypisane **prawidłowemu ruchowi** (np. `e2e4`).


| Prawdopodobieństwo (`p`) | Strata (`-log(p)`) | Interpretacja                             |
| ------------------------ | ------------------ | ----------------------------------------- |
| `p = 0.9` (90%)          | `-log(0.9) ≈ 0.1`  | Mały błąd – model prawie się nie pomylił. |
| `p = 0.5` (50%)          | `-log(0.5) ≈ 0.69` | Umiarkowany błąd.                         |
| `p = 0.1` (10%)          | `-log(0.1) ≈ 2.3`  | Duży błąd – model bardzo się pomylił.     |

###### Dlaczego `-log(p)`?
- Gdy `p` jest bliskie 1 (idealne), `-log(p)` jest bliskie 0 (brak kary).
- Gdy `p` jest bliskie 0 (błąd), `-log(p)` gwałtownie rośnie (duża kara).


### Cross-Entropy dla Wartości Pozycji (Regresja z Centipawn)

 Cel: Sprawić, by model **poprawnie szacował wartość pozycji** (np. "+1.5" – przewaga białych).

 Ponieważ dokładna wartość jest trudna do przewidzenia, dzielimy ją na **przedziały**, np.:  
`[-10, -5, -2, -1, -0.5, 0, +0.5, +1, +2, +5, +10]`.

- każdy przedział ma własne prawdopodobieństwo 
- Strata jest niska, jeśli najwyższe `p` jest w zawierającym przedziale ==prawdziwą wartość==


*Przykład*:
- **Przewidywania modelu:**
    - `+1.0`: 70%
    - `+1.5`: 25%
    - Inne biny: 5%
- **Strata jest niska**, bo prawdziwa wartość (`+1.23`) jest blisko bina `+1.0` (który ma wysokie `p`)
