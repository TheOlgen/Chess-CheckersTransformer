import torch
from typing import List, Tuple
import re

# piece_map_board: mapuje FEN do wartości liczbowych tensora
piece_map_board = {
    'w': 1,  # białe pionki
    'b': 2,  # czarne pionki
    'W': 3,  # białe damki
    'B': 4,  # czarne damki
    '.': 0  # puste pole
}

# reverse_piece_map_board: mapuje wartości liczbowe z powrtoem do FEN
reverse_piece_map_board = {
    1: 'w', 2: 'b', 3: 'W', 4: 'B',
    0: '.'  # Use '.' for empty for FEN string
}

def field_to_coords(field: int) -> Tuple[int, int]:
    """Zamienia numer pola (1–50) na współrzędne (rząd, kolumna) na planszy 10x10 (0-9)."""
    if not (1 <= field <= 50):
        raise ValueError(f"Niepoprawny numer pola: {field}")
    # W warcabach 10x10 pola numerowane są od lewej do prawej, od dołu do góry
    # Pole 1 jest na dole (row 9), na pozycji (9,1) lub (9,0) w zależności od parzystości rzędu.
    # Wiersz jest 9 - ((field - 1) // 5)
    # Kolumna jest (field - 1) % 5 * 2 + (row + 1) % 2

    row = 9 - (field - 1) // 5

    if row % 2 == 0:  # Rzędy (0, 2, ..., 8) odnaczaja 'top rows'
        col = (field - 1) % 5 * 2 + 1  # Dark squares - kolumny (1,3,5,7,9)
    else:  # Rzędy (1, 3, ..., 9) oznaczaja 'bottom rows'
        col = (field - 1) % 5 * 2  # Dark squares - kolumny (0,2,4,6,8)

    row_0_indexed = (field - 1) // 5
    if row_0_indexed % 2 == 0:
        col_0_indexed = ((field - 1) % 5) * 2 + 1
    else:
        col_0_indexed = ((field - 1) % 5) * 2

    return row_0_indexed, col_0_indexed


def coords_to_field(row: int, col: int) -> int:
    """Zamienia współrzędne (rząd, kolumna) na numer pola (1-50) na planszy 10x10 (0-9)."""
    if not (0 <= row < 10 and 0 <= col < 10):
        raise ValueError(f"Niepoprawne współrzędne: ({row}, {col})")
    if (row + col) % 2 == 0:  # Tylko ciemne pola
        return -1  # Oznacza nieprawidłowe pole

    field_number = row * 5
    if row % 2 == 0:  # Even rows (0, 2, ...) -> dark squares are 1, 3, 5, 7, 9
        field_number += (col // 2) + 1
    else:  # Odd rows (1, 3, ...) -> dark squares are 0, 2, 4, 6, 8
        field_number += (col // 2) + 1

    return field_number


def custom_pdn_fen_to_standard_fen(custom_pdn_fen: str) -> str:
    """
    Konwertuje custom PDN FEN ([FEN "B:W18,24,..."]) na standardowy FEN 10x10
    (np. "2b2b2b2b/b2b2b2b2/... b").
    """
    # Usunięcie '[FEN "' i '"]'
    content_match = re.match(r'\[FEN "(.*?)"\]', custom_pdn_fen)
    if not content_match:
        raise ValueError(f"Nieprawidłowy format custom PDN FEN: {custom_pdn_fen}")

    content = content_match.group(1)
    parts = content.split(':')

    if len(parts) < 1:
        raise ValueError("Nieprawidłowy format FEN - za mało części")

    turn_char = parts[0]
    turn = 'w' if turn_char == 'W' else 'b'

    board = [['.' for _ in range(10)] for _ in range(10)]

    for i in range(1, len(parts)):
        color_part = parts[i]
        if not color_part:
            continue

        color_code = color_part[0]
        piece_type = 'w' if color_code == 'W' else 'b'
        king_type = 'W' if color_code == 'W' else 'B'

        squares_str = color_part[1:]
        if not squares_str:
            continue

        square_numbers = squares_str.split(',')
        for sq_num_str in square_numbers:
            is_king = False
            if sq_num_str.startswith('K'):
                is_king = True
                sq_num = int(sq_num_str[1:])
            else:
                sq_num = int(sq_num_str)

            # Use field_to_coords to get 0-indexed (row, col)
            r, c = field_to_coords(sq_num)
            if 0 <= r < 10 and 0 <= c < 10:  # Ensure valid coordinates
                board[r][c] = king_type if is_king else piece_type

    # Konwersja planszy 2D do standardowego ciągu FEN
    fen_rows = []
    for r in range(10):
        fen_row_str = ""
        empty_count = 0
        for c in range(10):
            piece = board[r][c]
            if piece == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row_str += str(empty_count)
                    empty_count = 0
                fen_row_str += piece
        if empty_count > 0:
            fen_row_str += str(empty_count)
        fen_rows.append(fen_row_str)

    board_fen_part = "/".join(fen_rows)
    return f"{board_fen_part} {turn}"


def fen_to_board_tensor(standard_fen: str) -> torch.Tensor:
    """
    Konwertuje standardowy ciąg FEN dla warcabów 10x10 na tensor
    Tensor: 100 elementów dla pól + 1 element dla koloru ruchu (0=czarne, 1=białe).
    """
    parts = standard_fen.split()
    if len(parts) < 2:
        raise ValueError(f"Nieprawidłowy format FEN: {standard_fen}")

    board_part = parts[0]
    active_color_char = parts[1]  # 'w' or 'b'

    board_flat = [0] * 100  # 10x10 = 100 squares

    row_idx = 0
    col_idx = 0

    for char in board_part:
        if char == '/':
            row_idx += 1
            col_idx = 0
        elif char.isdigit():
            col_idx += int(char)
        else:
            # Map piece character to integer
            board_flat[row_idx * 10 + col_idx] = piece_map_board.get(char, 0)
            col_idx += 1

    active_color_val = 1 if active_color_char == 'w' else 0

    return torch.tensor(board_flat + [active_color_val], dtype=torch.long)


def board_tensor_to_fen(tensor: torch.Tensor) -> str:
    """
    Konwertuje tensor planszy na standardowy ciąg FEN dla warcabów 10x10.
    """
    board_flat = tensor[:100].tolist()
    active_color_val = tensor[100].item()
    active_color_char = 'w' if active_color_val == 1 else 'b'

    fen_rows = []
    for r in range(10):
        fen_row_str = ''
        empty_count = 0
        for c in range(10):
            idx = r * 10 + c
            piece_val = board_flat[idx]
            if piece_val == 0:  # Empty square
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row_str += str(empty_count)
                    empty_count = 0
                fen_row_str += reverse_piece_map_board.get(piece_val, '.')
        if empty_count > 0:
            fen_row_str += str(empty_count)
        fen_rows.append(fen_row_str)

    board_fen_part = '/'.join(fen_rows)
    return f"{board_fen_part} {active_color_char}"


def move_to_index(move_str: str) -> int:
    """
    Konwertuje ruch w formacie 'from_field-to_field' (np. '34-29') na pojedynczy indeks.
    Indeks: from_field * 51 + to_field (używając 51 jako bazę, bo pola są 1-50, więc 50 wartości).
    Maksymalny indeks: 50 * 51 + 50 = 2550 + 50 = 2600.
    """
    if '-' not in move_str:
        return 0

    try:
        from_field, to_field = map(int, move_str.split('-'))

        if not (1 <= from_field <= 50 and 1 <= to_field <= 50):
            raise ValueError(f"Nieprawidłowe numery pól w ruchu: {move_str}")

        return (from_field - 1) * 50 + (to_field - 1)
    except ValueError as e:
        print(f"Błąd konwersji ruchu '{move_str}' na indeks: {e}")
        return 0  #default/error index (Brak ruchów')

def index_to_move(index: int) -> str:
    """
    Konwertuje indeks z powrotem na ruch w formacie 'from_field-to_field'.
    """
    # Assuming index = (from_field - 1) * 50 + (to_field - 1)
    # from_field = index // 50 + 1
    # to_field = index % 50 + 1
    if not (0 <= index < 2500):  # Max index (50*50 - 1) = 2499
        return "N/A"  # Or handle as an error

    from_field = (index // 50) + 1
    to_field = (index % 50) + 1
    return f"{from_field}-{to_field}"


def checkersEvaluator(boards: torch.Tensor, preds: torch.Tensor):
    """
    Ocenia legalność przewidywanych ruchów.
    Wymaga Board z draughts, aby sprawdzić legalność ruchów.
    """
    import draughts  # Import draughts here if not imported globally
    illegal_count = 0

    for i in range(boards.shape[0]):
        board_tensor = boards[i]
        predicted_idx = preds[i].item()  # Get the predicted move index

        try:
            # 1. Konwertuj tensor planszy na standardowy FEN
            fen_string = board_tensor_to_fen(board_tensor)

            # 2. Utwórz obiekt Board z tego FEN
            board = draughts.Board(fen_string)

            # 3. Konwertuj przewidywany indeks na ruch (np. "34-29")
            predicted_move_str = index_to_move(predicted_idx)

            if predicted_move_str == "N/A":  # Handle invalid indices from model
                illegal_count += 1
                continue

            # 4. Sprawdź, czy ruch jest legalny na danej planszy
            # draughts.Board.parse_move() takes PDN move (e.g., '34-29')
            # and checks if it's legal given the current board state.

            legal_moves = board.legal_moves()
            is_legal = False
            for move in legal_moves:
                if move.pdn_move == predicted_move_str:
                    is_legal = True
                    break

            if not is_legal:
                illegal_count += 1
                # print(f"Nielegalny ruch dla FEN: {fen_string}, przewidziano: {predicted_move_str}")

        except Exception as e:
            # print(f"Błąd w checkersEvaluator dla tensora/predykcji {i}: {e}")
            illegal_count += 1  # Błąd przetwarzania = nielegalny ruch

    # print(f"Illegal moves in batch: {illegal_count}/{len(boards)}")
    return illegal_count


