import requests
import re
import copy
import csv

# === KONFIGURACJA ===
TOKEN = 'fGV1t6jvtfu118uO'
LIMIT = 100  # max 300

# Pobierz nazwę użytkownika z pliku
with open('username.txt', 'r') as f:
    USERNAME = f.read().strip()


# === PROCES 1: Pobieranie gier ===
def download_games():
    headers = {
        'Authorization': f'Bearer {TOKEN}',
        'Accept': 'application/x-chess-pgn'
    }

    url = f'https://lidraughts.org/api/games/user/{USERNAME}?max={LIMIT}&moves=true&tags=true&clocks=true'

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open('lidraughts_games_001.pdn', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f'✅ Zapisano {LIMIT} gier do "lidraughts_games_001.pdn"')
    else:
        print(f'❌ Błąd ({response.status_code}): {response.text}')
        exit()


# === PROCES 2: Ekstrakcja ruchów ===
def extract_moves(pdn_moves_text):
    no_comments = re.sub(r'\{[^}]*\}', '', pdn_moves_text)
    no_move_numbers = re.sub(r'\d+\.+', '', no_comments)
    tokens = no_move_numbers.split()
    moves = [t for t in tokens if re.match(r'^\d+[-x]\d+$', t)]
    return ' '.join(moves)


def parse_pdn_file():
    with open('lidraughts_games_001.pdn', 'r', encoding='utf-8') as f:
        content = f.read()

    games = content.strip().split('\n\n')
    moves_only_list = []

    for game in games:
        lines = game.splitlines()
        moves_lines = [line for line in lines if not line.startswith('[')]
        moves_text = ' '.join(moves_lines).strip()

        if moves_text:
            moves_only = extract_moves(moves_text)
            if moves_only:
                moves_only_list.append(moves_only)

    with open('moves_001.txt', 'w', encoding='utf-8') as f_out:
        for line in moves_only_list:
            f_out.write(line + '\n')


# === PROCES 3: Konwersja do FEN ===
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


def convert_moves():
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
                out.write("dupa")


# === PROCES 4: Generowanie CSV ===
def parse_simple_pdn(pdn_str):
    try:
        # Podziel na części i sprawdź czy mamy przynajmniej 2 elementy
        parts = pdn_str.strip().split()
        if len(parts) < 2:
            raise ValueError("Nieprawidłowy format FEN - za mało części")

        # Ostatni element to tura, reszta to plansza
        turn = parts[-1]
        if turn not in ['w', 'b']:
            raise ValueError(f"Nieprawidłowa tura: {turn}")

        rows_str = ' '.join(parts[:-1])
        rows = rows_str.split("/")

        if len(rows) != 10:
            raise ValueError(f"Nieprawidłowa liczba wierszy: {len(rows)} (oczekiwano 10)")

        board = []
        for row in rows:
            parsed_row = []
            for token in re.findall(r'\d+|[wbWB]', row):
                if token.isdigit():
                    parsed_row.extend(['.'] * int(token))
                else:
                    parsed_row.append(token)

            if len(parsed_row) != 10:
                raise ValueError(f"Nieprawidłowa liczba kolumn: {len(parsed_row)} (oczekiwano 10)")

            board.append(parsed_row)

        return board, turn

    except Exception as e:
        print(f"Błąd parsowania FEN: '{pdn_str[:50]}...' - {e}")
        raise


def is_inside(x, y):
    return 0 <= x < 10 and 0 <= y < 10


def is_opponent(piece, player):
    if piece == '.':
        return False
    if player in ['w', 'W']:
        return piece in ['b', 'B']
    else:
        return piece in ['w', 'W']


def generate_moves(board, player):
    moves = []
    is_king = lambda p: p in ['W', 'B']
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(10):
        for j in range(10):
            piece = board[i][j]
            if piece.lower() != player:
                continue
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if is_inside(ni, nj) and board[ni][nj] == '.':
                    if is_king(piece) or (player == 'w' and dx == -1) or (player == 'b' and dx == 1):
                        moves.append(((i, j), (ni, nj)))
                bi, bj = i + dx, j + dy
                ti, tj = i + 2 * dx, j + 2 * dy
                if is_inside(ti, tj) and is_opponent(board[bi][bj], player) and board[ti][tj] == '.':
                    if is_king(piece) or (player == 'w' and dx == -1) or (player == 'b' and dx == 1):
                        moves.append(((i, j), (ti, tj)))
    return moves


def apply_move(board, move):
    (i1, j1), (i2, j2) = move
    new_board = copy.deepcopy(board)
    piece = new_board[i1][j1]
    new_board[i1][j1] = '.'
    new_board[i2][j2] = piece

    if abs(i2 - i1) == 2:
        mi, mj = (i1 + i2) // 2, (j1 + j2) // 2
        new_board[mi][mj] = '.'

    if piece == 'w' and i2 == 0:
        new_board[i2][j2] = 'W'
    elif piece == 'b' and i2 == 9:
        new_board[i2][j2] = 'B'

    return new_board


def evaluate_board(board, player):
    score = 0
    for row in board:
        for p in row:
            if p == player:
                score += 1
            elif p == player.upper():
                score += 2
            elif p.lower() != '.' and p.lower() != player:
                score -= 1 if p.islower() else 2
    return score


def best_move(board, player):
    possible_moves = generate_moves(board, player)
    best_score = -float('inf')
    best = None
    for move in possible_moves:
        new_board = apply_move(board, move)
        score = evaluate_board(new_board, player)
        if score > best_score:
            best_score = score
            best = move
    return best


def position_to_square(row, col):
    if (row + col) % 2 == 0:
        raise ValueError(f"Pole ({row}, {col}) nie jest czarne.")
    return (row * 5) + (col // 2) + 1


def generate_csv():
    with open("converted_moves.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with open("Database/evaluation_001.csv", mode="w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["FEN", "najlepszy_ruch"])

        for i, line in enumerate(lines, 1):
            try:
                board, player = parse_simple_pdn(line)
                move = best_move(board, player)
                if move is not None:
                    from_sq = position_to_square(*move[0])
                    to_sq = position_to_square(*move[1])
                    best_move_str = f"{from_sq}-{to_sq}"
                    writer.writerow([line, best_move_str])
                else:
                    print(f"Nie znaleziono ruchu dla linii {i}")
            except Exception as e:
                print(f"Błąd w linii {i}: {str(e)}")
                continue


# === GŁÓWNY PROGRAM ===
def main():
    print(f"Pobieranie gier dla użytkownika: {USERNAME}")
    download_games()

    print("\nEkstrakcja ruchów...")
    parse_pdn_file()

    print("\nKonwersja do FEN...")
    with open("moves_001.txt", "r") as f:
        print(f"Liczba partii do przetworzenia: {len(f.readlines())}")

    convert_moves()

    print("\nGenerowanie pliku CSV...")
    with open("converted_moves.txt", "r") as f:
        print(f"Liczba pozycji FEN: {len(f.readlines())}")

    generate_csv()
    print("\nGotowe! Wynik zapisano w evaluation_001.csv")


if __name__ == "__main__":
    main()