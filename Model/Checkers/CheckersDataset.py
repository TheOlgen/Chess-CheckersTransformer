from typing import List

import torch
from torch.utils.data import Dataset, IterableDataset
import draughts
from Checkers.Database.SQL_checkers import get_positions


class CheckersDataset(Dataset):
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



class CheckersStreamDataset(IterableDataset):
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


#nwm czy tu git
def square_to_index(square):
    if square == '-':
        return 10
    file = ord(square[0]) - ord('a')
    rank = int(square[1]) - 1
    return rank * 10 + file


def fen_to_board(fen: str) -> torch.Tensor:
    piece_map = {
        'w': 1,  # białe pionki
        'b': 2,  # czarne pionki
        'W': 3,  # białe damki
        'B': 4,  # czarne damki
        '.': 0  # puste pole
    }

    # Rozdzielamy część z planszą i informacje o ruchu
    parts = fen.split()
    board_part = parts[0]
    active_color = parts[1][0]  # 'w' lub 'b' przed przecinkiem

    # Inicjalizacja pustej planszy 10x10 (100 pól)
    board = [0] * 100

    row = 0
    col = 0

    for char in board_part:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            index = row * 10 + col
            board[index] = piece_map.get(char, 0)
            col += 1

    # Dodajemy informację o tym, czyj jest ruch (0 - czarne, 1 - białe)
    color_value = 1 if active_color == 'w' else 0

    return torch.tensor(board + [color_value], dtype=torch.long)


def board_tensor_to_fen(tensor: torch.Tensor) -> str:
    reverse_piece_map = {
        1: 'w', 2: 'b', 3: 'W', 4: 'B',
        0: ' '
    }

    board = tensor[:100].tolist()
    active_color = 'w' if tensor[100].item() == 1 else 'b'

    fen_rows = []
    for row in range(10):
        fen_row = ''
        empty = 0
        for col in range(10):
            index = row * 10 + col
            piece = board[index]
            if piece == 0:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += reverse_piece_map[piece]
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    position_part = '/'.join(fen_rows)
    return f"{position_part} {active_color}"


def checkersEvaluator(boards, preds):
    illegal = 0

    for board_tensor, pred_idx in zip(boards, preds):
        # Konwersja tensora do Twojego formatu FEN
        fen = board_tensor_to_fen(board_tensor)

        # Pobranie części z planszą (pomijamy kolor i ruch)
        board_part = fen.split()[0]

        # Sprawdzenie poprawności ruchu
        move_idx = pred_idx.item()
        from_sq = move_idx // 100
        to_sq = move_idx % 100

        # Tutaj należy dodać logikę sprawdzania legalności ruchu
        # W Twoim przypadku możesz potrzebować własnej implementacji
        # ponieważ standardowe biblioteki mogą nie obsługiwać Twojego formatu

        # Przykładowe proste sprawdzenie:
        # 1. Czy pola są w zakresie 1-100
        # 2. Czy z pola 'from' jest pionek gracza
        # 3. Czy pole 'to' jest puste
        # (pełna implementacja wymaga zasad ruchu w warcabach)

        if not (1 <= from_sq <= 100) or not (1 <= to_sq <= 100):
            illegal += 1
            continue

        # Tutaj dodaj bardziej szczegółowe sprawdzenie zasad gry
        # ...

    print(f"Illegal moves: {illegal}/{len(boards)}")
    return illegal


def move_to_index(move_str: str) -> int:
    # Przykład: "41-46" → (41, 46)
    from_sq, to_sq = map(int, move_str.split('-'))
    return from_sq * 100 + to_sq
