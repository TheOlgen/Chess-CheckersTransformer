# Chess Without Search – A Transformer-Based Approach to Chess and Draughts

This project explores the application of a **decoder-only Transformer architecture** (similar to GPT models) to predict moves in **chess** and **draughts (checkers)** without relying on search-based methods like those used in traditional engines (e.g., Stockfish).

> 📄 For more in-depth technical details (in Polish), see: [`docs/sprawozdanie.pdf`](docs/sprawozdanie.pdf)

## 🚀 Project Overview

Instead of building a search tree to find the best move, we trained a Transformer model to predict the next move **based solely on the current board position**.

The model receives a FEN-like representation of a game state and is trained to output the move that a traditional engine (Stockfish or Scan Engine) would suggest.

## 🛠 Implementation Highlights

- **Language:** Python 3.10  
- **Libraries:** PyTorch, PyTorch Lightning, python-chess, pydraughts  
- **Data Source:**  
  - Chess: Lichess PGN → converted to FEN  
  - Draughts: LiDraughts custom format  

- **Evaluation Engines:**  
  - Chess: Stockfish  
  - Draughts: Scan Engine (via `pydraughts`)

- **Storage:** SQLite database for storing board positions and moves  
- **Training:**  
  - `batch_size = 64`, `d_model = 512`  
  - 20 epochs, learning rate decay, early stopping  

## 🤖 Model Architecture

- Decoder-only Transformer (like GPT)
- 6 Transformer blocks with:
  - Multi-head masked self-attention (8 heads)
  - Feed-forward network with ReLU
  - Layer Normalization and Dropout (0.1)
- Output logits over a large tokenized move vocabulary

## 📊 Evaluation Metrics

- **Accuracy** – top-1 match with engine suggestion  
- **Cross-entropy loss**  
- **Illegal Moves Count** – whether predicted move is legal  

The model occasionally predicted correct moves but frequently generated illegal ones. Key limitations:
- Too few training samples vs. model complexity  
- Overly large output space (all legal and illegal moves)  
- Hardware constraints and limited training time  


## 🔮 Future Work

- Reduce output space using legal move masks  
- Smaller/lighter models or pretrained embeddings  
- Visualization of attention on the board  
- Improve input/output representations  

## 👥 Authors

- Kacper Mikołajuk  
- Natalia Dembkowska  
- Natalia Sekula  
- Olga Rodziewicz  
- Patryk Lewandowski  

## 🔗 Resources

- [Grandmaster-Level Chess Without Search (arXiv)](https://arxiv.org/html/2402.04494v1)  
- [Stockfish Engine](https://stockfishchess.org/)  
- [Lichess Dataset on Kaggle](https://www.kaggle.com/datasets/arevel/chess-games/data)  
- [LiDraughts](https://lidraughts.org/)  
- [Chess Transformers GitHub](https://github.com/sgrvinod/chess-transformers)

---

> Developed at Gdańsk University of Technology as part of a course project.
