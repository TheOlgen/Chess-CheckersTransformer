from typing import List

import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, fens: List[str], best_moves: List[int]):
        assert len(fens) == len(best_moves)
        self.fens = fens
        self.best_moves = best_moves

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        move_idx = torch.tensor(self.best_moves[idx], dtype=torch.long)
        board_tensor = self.fen_to_board(fen)
        return board_tensor, move_idx

    def fen_to_board(self, fen):
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




