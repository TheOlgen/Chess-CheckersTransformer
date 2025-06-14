from typing import List

import torch
from torch.utils.data import Dataset, IterableDataset
import chess
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


# def fen_to_board(fen: str) -> torch.Tensor:
#     piece_map = {
#         'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1,
#         'k': 12, 'q': 11, 'r': 10, 'b': 9, 'n': 8, 'p': 7
#     }
#     rows = fen.split(' ')[0].split('/')
#     board = []
#     for row in rows:
#         for ch in row:
#             if ch.isdigit():
#                 board.extend([0] * int(ch))
#             else:
#                 board.append(piece_map.get(ch, 0))
#     #TODO: DODAJ pozostałe informacje z fena w ostatniej dodatkowej linijce tensora
#     return torch.tensor(board, dtype=torch.long)

    # En passant — zakodujemy jako numer pola od 0 do 63, lub 64 jeśli brak
def square_to_index(square):
    if square == '-':
        return 64
    file = ord(square[0]) - ord('a')
    rank = int(square[1]) - 1
    return rank * 8 + file

def fen_to_board(fen: str) -> torch.Tensor:
    piece_map = {
        'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1,
        'k': 12, 'q': 11, 'r': 10, 'b': 9, 'n': 8, 'p': 7
    }

    parts = fen.split()
    rows = parts[0].split('/')
    board = []

    for row in rows:
        for ch in row:
            if ch.isdigit():
                board.extend([0] * int(ch))
            else:
                board.append(piece_map.get(ch, 0))

    # Kodowanie pozostałych informacji jako liczby całkowite:
    # Aktywny kolor: 0 = b, 1 = w
    active_color = 1 if parts[1] == 'w' else 0

    # Roszady: 4 bity - K, Q, k, q
    castling_rights = 0
    if 'K' in parts[2]: castling_rights |= 1 << 3
    if 'Q' in parts[2]: castling_rights |= 1 << 2
    if 'k' in parts[2]: castling_rights |= 1 << 1
    if 'q' in parts[2]: castling_rights |= 1 << 0


    en_passant = square_to_index(parts[3])

    # Półruchy i pełne ruchy jako liczby
    halfmove_clock = int(parts[4])
    fullmove_number = int(parts[5])

    # Dodajemy te dane jako dodatkowe 5 wartości:
    extra = [active_color, castling_rights, en_passant, halfmove_clock, fullmove_number]

    full_tensor = board + extra
    return torch.tensor(full_tensor, dtype=torch.long)


def board_tensor_to_fen(tensor: torch.Tensor) -> str:
    piece_map = {
        6: 'K', 5: 'Q', 4: 'R', 3: 'B', 2: 'N', 1: 'P',
        12: 'k', 11: 'q', 10: 'r', 9: 'b', 8: 'n', 7: 'p'
    }

    board = tensor[:64]
    extra = tensor[64:]

    # Odtwarzanie układu bierek
    fen_rows = []
    for i in range(8):
        row = board[i * 8:(i + 1) * 8]
        fen_row = ''
        empty = 0
        for square in row:
            val = square.item()
            if val == 0:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece_map.get(val, '?')
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    position_part = '/'.join(fen_rows)

    # Odtwarzanie pozostałych części FEN
    active_color = 'w' if extra[0].item() == 1 else 'b'

    castling_bits = extra[1].item()
    castling_rights = ''
    if castling_bits & (1 << 3): castling_rights += 'K'
    if castling_bits & (1 << 2): castling_rights += 'Q'
    if castling_bits & (1 << 1): castling_rights += 'k'
    if castling_bits & (1 << 0): castling_rights += 'q'
    if castling_rights == '':
        castling_rights = '-'

    en_passant_index = extra[2].item()
    if en_passant_index == 64:
        en_passant = '-'
    else:
        file = chr((en_passant_index % 8) + ord('a'))
        rank = str((en_passant_index // 8) + 1)
        en_passant = file + rank

    halfmove_clock = str(extra[3].item())
    fullmove_number = str(extra[4].item())

    # Składanie pełnego FEN-a
    fen = f"{position_part} {active_color} {castling_rights} {en_passant} {halfmove_clock} {fullmove_number}"
    return fen


def chessEvaluator(boards, preds):
    illegal = 0

    for board_tensor, pred_idx in zip(boards, preds):
        fen = board_tensor_to_fen(board_tensor)
        board = chess.Board(fen)

        # Rozkodowanie indeksu ruchu na from-to (jeśli pred to int z zakresu 0-4095)
        move_idx = pred_idx.item()
        from_square = move_idx // 64
        to_square = move_idx % 64
        move = chess.Move(from_square, to_square)

        if move not in board.legal_moves:
            illegal += 1
    print("Illegal:", illegal, type(illegal))
    return illegal


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
