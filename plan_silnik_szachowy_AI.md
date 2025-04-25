
# Plan projektu: Silnik Szachowy AI (Python)
*Data wygenerowania: 23 kwietnia 2025*

## Założenia wysokiego poziomu  
- **Minimax / α‑β** wykorzystywany *wyłącznie* do wstępnego etykietowania pozycji podczas uczenia (Supervised Learning).  
- Model po fazie pre‑training *gra bez wyszukiwania* – ruch wybiera sama sieć (policy + value).  
- W późniejszym rozwoju można dołączyć wyszukiwanie (MCTS, hybryda), ale nie jest ono wymagane do pierwszej wersji MVP.  

---

## Harmonogram (wersja skrócona)

| Faza | Zakres | Czas (tyg.) | Rezultat |
|------|--------|------------|----------|
| 0 | Kick‑off, CI/CD, środowisko | 1‑2 | repo, benchmark baseline |
| 1 | Szkielet silnika + prosty α‑β do generacji danych | 3‑4 | silnik testowy, perft ok |
| 2 | Pipeline danych (PGN → FEN, bufor) | 3‑6 | 200 M stanowisk |
| 3 | Eksperymenty sieci (ResNet, Transformer, NNUE) | 5‑10 | pierwsze checkpointy |
| 4 | Pętla uczenia RL (self‑play bez minimax) | 8‑20 | sieć > 2600 ELO blitz |
| 5 | Optymalizacje wydajności | 18‑26 | INT8, GPU graphs |
| 6 | Walidacja i benchmark | 22‑28 | gauntlet 1 000 gier |
| 7 | Deployment UCI / API | 24‑30 | obrazy Docker | 


---

## Kluczowe komponenty

### 1. Reprezentacja pozycji
- **python‑chess** do generowania ruchów, FEN, legalności.  
- Bufor bitboardów `uint64[12]` + metadata (side to move, roszady, ep‑square).  
- Inkrementalny *hash* Zobrista do szybkiego cofania ruchów.

### 2. Generacja danych z minimax
1. Uruchom własny moduł α‑β (głębokość 6, LMR, null‑move).  
2. Dla każdej pozycji zapisz:  
   - wektor *policy* (1‑hot najlepszy ruch z wyszukiwania),  
   - skalary *value* (wynik partii lub eval α‑β normalizowany do [‑1, 1]).  
3. Składowanie w Parquet (kolumny: fen, policy_idx, value, meta).

### 3. Architektura sieci  
- **Input**: tensor [12 × 8 × 8] bitboardów + 6 kanałów statusu.  
- **Backbone**:  
  - wariant **Mini‑Transformer** 8 warstw, 256 d‑model, Rotary Pos Emb,  
  - lub **ResNet‑20** (benchmark).  
- **Głowy**:  
  - **Policy**: conv 1×1 → softmax(4672 ruchów kodowanych UCI).  
  - **Value**: global mean pool → lin → tanh.  
- **Parametry** ≈ 35 M (Transformer) / 7 M (ResNet).

### 4. Strategia uczenia  
1. **SL Stage** – 20‑40 epok na zbiorze α‑β, loss = CE(policy)+MSE(value).  
2. **RL Stage** – self‑play *bez szukania*, alg. PPO lub AlphaZero‑policy‑improvement:  
   - Każdy aktor symuluje partię, wybiera ruch ~ policy^(1/T).  
   - Batchy = 4096 pozycji; aktualizacja co N kroków.  
   - Regularyzacja KL do starej sieci, temp. annealing.  

### 5. Inference (silnik gry)
- Dla pozycji P:  
  1. ```moves = legal_moves(P)```  
  2. odpal forward pass sieci → vector policy, skalar value  
  3. wybierz `argmax(policy)` (lub top‑k sampling przy tempo > 1 min).  
- Latencja GPU FP16 ≈ 0.1 ms; CPU INT8 ≈ 0.8 ms.

### 6. Benchmark i walidacja  
| Test | Cel | Kryterium zaliczenia |
|------|-----|----------------------|
| Gauntlet vs Stockfish (depth 6) | stabilność ELO | ±5 ELO/commit |
| Lichess STS | taktyka | > 70 % poprawnych |
| 100k self‑play | leak/memory | brak crashy |

### 7. Deployment  
- **UCI bridge** w Python.  
- Kontener `Dockerfile` z modelem ONNX + skróconym serwerem.  
- Alternatywa: TorchScript + libtorch C++.

---

## Milestone po 12 tygodniach
- Sieć ResNet‑20 przewyższa minimax depth 4 o ≥ 150 ELO (tempo 1 + 0.5).  
- Gra bierze ⩾ 45 % punktów vs Stockfish 15 przy depth 6.  
- Build CPU‑INT8 < 50 MB, latencja < 1 ms/move.

---


