# Szachy bez przeszukiwania – transformer do gry w szachy i warcaby

Projekt realizowany na Politechnice Gdańskiej miał na celu zbadanie możliwości zastosowania architektury **transformera** do uczenia gry w szachy i warcaby bez użycia klasycznego przeszukiwania drzew decyzyjnych.

## ✨ Opis projektu

W przeciwieństwie do tradycyjnych silników (np. Stockfish), które opierają się na głębokim przeszukiwaniu drzewa ruchów, nasz model uczy się generować kolejne ruchy wyłącznie na podstawie bieżącej pozycji na planszy, przy użyciu architektury **transformera typu decoder-only (GPT-like)**.

Projekt obejmuje:
- Tokenizację plansz szachowych (FEN) i warcabowych.
- Przetwarzanie i ocenę pozycji przy pomocy silników Stockfish i Scan Engine.
- Trening modelu transformera w PyTorch Lightning.
- Ewaluację dokładności przewidywanych ruchów i ich legalności.

## 🧠 Architektura modelu

- Architektura typu `decoder-only`, inspirowana GPT.
- Warstwa embeddingu + kodowanie pozycyjne.
- 6 bloków transformera (8 głowic, FFN z ReLU, LayerNorm, Dropout).
- CrossEntropyLoss do porównania przewidywań z ruchami silnika.

## 🛠️ Implementacja

Główne komponenty:
- Python 3.10
- PyTorch, PyTorch Lightning
- `python-chess`, `pydraughts`, `sqlite3`
- Trening: batch_size = 64, d_model = 512, AdamW

Zbiór danych:
- Szachy: dane z Lichess w formacie PGN → FEN
- Warcaby: dane z LiDraughts
- Pozycje oceniane silnikami i zapisane w SQLite

## 📊 Ewaluacja

Model oceniano na podstawie:
- **Accuracy** – trafność przewidywanego ruchu.
- **Cross-entropy loss** – funkcja straty.
- **Illegal Moves** – liczba nielegalnych przewidywań.

Wyniki były niezadowalające — model trafiał w ruchy silnika sporadycznie i często przewidywał nielegalne posunięcia. Główne problemy to:
- Za mała ilość danych i ograniczenia sprzętowe.
- Zbyt wysoka liczba parametrów (d_model=512).
- Zbyt szeroka przestrzeń wyjściowa (wszystkie możliwe ruchy).


## 🔮 Możliwe usprawnienia

- Redukcja przestrzeni wyjściowej (np. maskowanie legalnych ruchów).
- Uproszczona reprezentacja planszy jako obraz.
- Zastosowanie mniejszych modeli lub technik transfer learningu.
- Lepsze dostrojenie hiperparametrów i dłuższy trening.

## 👥 Autorzy

- Kacper Mikołajuk  
- Natalia Dembkowska  
- Natalia Sekula  
- Olga Rodziewicz  
- Patryk Lewandowski  

## 📚 Źródła

- [Grandmaster-Level Chess Without Search](https://arxiv.org/html/2402.04494v1)
- [Stockfish Chess Engine](https://stockfishchess.org/)
- [Lichess Chess Data (Kaggle)](https://www.kaggle.com/datasets/arevel/chess-games/data)
- [LiDraughts](https://lidraughts.org/)
- [Chess Transformers repo](https://github.com/sgrvinod/chess-transformers)

---

> Projekt zrealizowany w ramach zajęć na Politechnice Gdańskiej.
