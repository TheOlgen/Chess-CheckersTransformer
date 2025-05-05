**Zanikanie gradientów** (_vanishing gradients_) to problem w uczeniu głębokich sieci neuronowych, gdzie **informacja (gradient) "znika"** podczas propagacji wstecznej (backpropagacji), przez co wagi w początkowych warstwach uczą się bardzo wolno lub wcale.

Gradient - pochodna funkcji straty względem wag modelu. Mówi nam, **jak mocno i w jakim kierunku** należy dostosować wagi, aby zmniejszyć błąd.

Backpropagacja -  Model wykonuje predykcję (np. ocenia pozycję szachową). Następnie liczy stratę (np. różnicę między przewidzianą a prawdziwą wartością). Propaguje gradient wstecz - od ostatniej warstwy do pierwszej, aby zaktualizować wagi



##### Ale czemu te gradienty zanikają?

- W głębokich sieciach (np. transformatorach) gradient jest **mnożony przez wiele małych liczb** (pochodnych funkcji aktywacji, np. sigmoid).
- Jeśli pochodne są <1, ich iloczyn **staje się bliski zeru** → gradient "znika".
- **Skutek:** Wagi w początkowych warstwach **prawie się nie aktualizują** → model przestaje się uczyć.