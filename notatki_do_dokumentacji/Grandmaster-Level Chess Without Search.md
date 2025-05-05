
Dla fanów - to na razie nie dokumentacja, tylko notatki
[[skrót]]
 [[Wstęp]]
 [[Proces treningu modelu]]

[[Przygotowanie danych treningowych]]
[[Model (transformator dekoderowy)]]
[[Przewidywania i strategie]]
[[Metryki ewaluacyjne]]




# Centipawn

To sposób na określenie wartości pozycji w szachach za pomocą liczb. Wartości bazowe figur (przybliżone): 
- Pion = 100 centy-pionków (1,00)
- Skoczek = ~300 (3,00)
- Goniec = ~300 (3,00)
- Wieża = ~500 (5,00)
- Hetman = ~900 (9,00)
- Król = ==bezcenny== (ale w końcówce ma "wartość" ok. 400-500)
Dodatkowo też są inne czynniki tj. 
- pozycja : czy pionki podbijają centrum (lepiej), czy może są na skraju (gorzej)
- struktura pionków : czy pionki izolowane, zdublowane 
- aktywność figur 
- bezpieczeństwo króla
- kontrola centrum
Interpretacja oceny:
- **+1,00**: Białe mają przewagę równą jednemu pionowi.
- **-0,50**: Czarne mają przewagę równą pół piona.
- **+3,20**: Białe mają przewagę ~3 pionów i 20 centów (np. wieża za skoczka).
- **0,00**: Pozycja jest zrównoważona.


