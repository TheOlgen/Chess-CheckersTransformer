import torch
from Model.Model import ChessTransformer

def fen_to_board(fen):
    # Mapa figur na liczby (dla szachów)
    piece_map = {
        'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1,  # Białe figury
        'k': 12, 'q': 11, 'r': 10, 'b': 9, 'n': 8, 'p': 7  # Czarne figury
    }
    rows = fen.split(' ')[0].split('/')
    board = []

    for row in rows:
        for char in row:
            if char.isdigit():  # Puste pola (np. 3 oznacza 3 puste pola)
                board.extend([0] * int(char))  # Dodajemy odpowiednią liczbę zer
            else:  # Figury
                board.append(piece_map.get(char, 0))  # Wstawiamy odpowiednią figurę

    return torch.tensor(board, dtype=torch.int64)

fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
board = fen_to_board(fen)
model = ChessTransformer(d_model=512, max_len=64, num_moves=4096)
print(board)
best_move = model.predict_move(board)
print(f"Najlepszy ruch: {best_move}")
