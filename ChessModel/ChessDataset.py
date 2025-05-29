from typing import List

import torch
from torch.utils.data import Dataset, IterableDataset

from Chess.Database.SQL_chess import get_positions


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
        board_tensor = fen_to_board(fen)
        return board_tensor, move_idx

    # ChessDataset/streaming_dataset.py

    import torch

    # def fen_to_board(self, fen):
    #     piece_map = {
    #         'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1,  # Białe figury
    #         'k': 12, 'q': 11, 'r': 10, 'b': 9, 'n': 8, 'p': 7  # Czarne figury
    #     }
    #     rows = fen.split(' ')[0].split('/')
    #     board = []
    #
    #     for row in rows:
    #         for char in row:
    #             if char.isdigit():  # Puste pola (np. 3 oznacza 3 puste pola)
    #                 board.extend([0] * int(char))  # Dodajemy odpowiednią liczbę zer
    #             else:  # Figury
    #                 board.append(piece_map.get(char, 0))  # Wstawiamy odpowiednią figurę
    #
    #     return torch.tensor(board, dtype=torch.int64)


class ChessStreamDataset(IterableDataset):
    def __init__(self, chunk_size: int = 1000):
        # chunk_size: ile rekordów pobieramy na raz z bazy
        self.chunk_size = chunk_size

    def __iter__(self):
        while True:
            batch = get_positions(self.chunk_size)
            if not batch:
                break

            for fen, best_move in batch:
                board_tensor = fen_to_board(fen)
                move_idx = move_to_index(best_move)
                yield board_tensor, torch.tensor(move_idx, dtype=torch.long)


def fen_to_board(fen: str) -> torch.Tensor:
    piece_map = {
        'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1,
        'k': 12, 'q': 11, 'r': 10, 'b': 9, 'n': 8, 'p': 7
    }
    rows = fen.split(' ')[0].split('/')
    board = []
    for row in rows:
        for ch in row:
            if ch.isdigit():
                board.extend([0] * int(ch))
            else:
                board.append(piece_map.get(ch, 0))
    return torch.tensor(board, dtype=torch.long)


def move_to_index(move: str) -> int:
    #Zamienia ruch w notacji 'a1a4' na indeks [0..4095],

    if len(move) != 4:
        raise ValueError(f"Nieoczekiwany format ruchu: {move!r}, oczekiwany e.g. 'e2e4'")
    file_s, rank_s, file_e, rank_e = move[0], move[1], move[2], move[3]

    # zamiana kolumny 'a'–'h' na 0–7
    f_start = ord(file_s) - ord('a')
    f_end = ord(file_e) - ord('a')
    # zamiana rzędu '1'–'8' na indeks 0–7
    r_start = int(rank_s)
    r_end = int(rank_e)

    # odwrotność idx_to_coord:
    start_idx = (8 - r_start) * 8 + f_start
    end_idx = (8 - r_end) * 8 + f_end

    return start_idx * 64 + end_idx
