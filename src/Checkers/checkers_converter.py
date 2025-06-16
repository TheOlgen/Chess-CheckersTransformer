def field_to_coords(field):
    """Zamienia numer pola (1–50) na współrzędne (rząd, kolumna) na planszy 10x10"""
    if not (1 <= field <= 50):
        raise ValueError(f"Niepoprawny numer pola: {field}")
    row = 9 - (field - 1) // 5
    col = ((field - 1) % 5) * 2 + ((row + 1) % 2)
    return row, col

def initial_board():
    """Tworzy początkową planszę 10×10 z pionkami białymi (w) i czarnymi (b)"""
    board = [["." for _ in range(10)] for _ in range(10)]
    # Czarny: pola 1–20 (góra)
    for i in range(1, 21):
        r, c = field_to_coords(i)
        board[r][c] = "b"
    # Biały: pola 31–50 (dół)
    for i in range(31, 51):
        r, c = field_to_coords(i)
        board[r][c] = "w"
    return board

def to_fen(board, turn):
    fen_rows = []
    for row in board:
        empty = 0
        fen_row = ""
        for cell in row:
            if cell == ".":
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " " + turn

def move_piece(board, from_field, to_field):
    r1, c1 = field_to_coords(from_field)
    r2, c2 = field_to_coords(to_field)
    piece = board[r1][c1]
    board[r1][c1] = "."
    board[r2][c2] = piece

def apply_move(board, move):
    if 'x' in move:
        squares = list(map(int, move.split('x')))
        for i in range(len(squares) - 1):
            move_piece(board, squares[i], squares[i + 1])
            # Usuwanie zbitego pionka – pomiędzy ruchami
            r1, c1 = field_to_coords(squares[i])
            r2, c2 = field_to_coords(squares[i + 1])
            jr, jc = (r1 + r2) // 2, (c1 + c2) // 2
            board[jr][jc] = "."
    else:
        squares = list(map(int, move.split('-')))
        for i in range(len(squares) - 1):
            move_piece(board, squares[i], squares[i + 1])

def parse_game(moves_str):
    import re
    moves = re.findall(r"\d+[\-x]\d+(?:[\-x]\d+)*", moves_str)
    board = initial_board()
    turn = 'w'
    positions = []

    for move in moves:
        apply_move(board, move)
        turn = 'b' if turn == 'w' else 'w'
        fen = to_fen(board, turn)
        positions.append(fen)
    return positions

def my_main():
    input_file = "moves_001.txt"
    output_file = "converted_moves.txt"

    with open(input_file, "r") as f:
        games = f.readlines()

    with open(output_file, "w") as out:
        for idx, game in enumerate(games):
            game = game.strip()
            if not game:
                continue
            try:
                positions = parse_game(game)
                out.write(f"Game {idx + 1}:\n")
                for i, pos in enumerate(positions):
                    out.write(f"{pos}\n")
                out.write("\n")
            except Exception as e:
                #out.write(f"Game {idx + 1} — BŁĄD: {e}\n\n")
                out.write("")

if __name__ == "__main__":
    my_main()
