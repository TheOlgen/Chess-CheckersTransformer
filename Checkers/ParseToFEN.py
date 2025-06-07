import requests
import re
import copy

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


# === PROCES 3: Konwersja do PDN FEN ===
def field_to_coords(field):
    """Zamienia numer pola (1–50) na współrzędne (rząd, kolumna) na planszy 10x10"""
    if not (1 <= field <= 50):
        raise ValueError(f"Niepoprawny numer pola: {field}")
    row = 9 - (field - 1) // 5
    col = ((field - 1) % 5) * 2 + ((row + 1) % 2)
    return row, col


def coords_to_field(row, col):
    """Zamienia współrzędne (rząd, kolumna) na numer pola (1-50) na planszy 10x10"""
    if not (0 <= row < 10 and 0 <= col < 10):
        raise ValueError(f"Niepoprawne współrzędne: ({row}, {col})")
    if (row + col) % 2 == 0:  # Tylko ciemne pola są używane w warcabach
        return None

    # Oblicz podstawowy numer pola dla wiersza
    base_field = (9 - row) * 5

    # Dostosuj na podstawie kolumny i parzystości wiersza
    if row % 2 == 0:  # Parzyste wiersze (0, 2, ..., 8)
        field = base_field + (col // 2) + 1
    else:  # Nieparzyste wiersze (1, 3, ..., 9)
        field = base_field + (col // 2) + 1
    return field


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


def to_pdn_fen(board, turn):
    white_pieces = []
    black_pieces = []

    for r in range(10):
        for c in range(10):
            piece = board[r][c]
            field_number = coords_to_field(r, c)
            if field_number is None:  # Pomijaj jasne pola
                continue

            if piece == 'w':
                white_pieces.append(str(field_number))
            elif piece == 'W':
                white_pieces.append(f'K{field_number}')
            elif piece == 'b':
                black_pieces.append(str(field_number))
            elif piece == 'B':
                black_pieces.append(f'K{field_number}')

    turn_char = 'W' if turn == 'w' else 'B'

    # Sortuj dla spójnego wyjścia
    white_pieces.sort(key=lambda x: int(x.replace('K', '')))
    black_pieces.sort(key=lambda x: int(x.replace('K', '')))

    white_str = f"W{','.join(white_pieces)}" if white_pieces else ""
    black_str = f"B{','.join(black_pieces)}" if black_pieces else ""

    # Połącz dwukropkiem, obsługując przypadki, gdy jeden kolor nie ma figur
    if white_str and black_str:
        return f'[FEN "{turn_char}:{white_str}:{black_str}"]'
    elif white_str:
        return f'[FEN "{turn_char}:{white_str}"]'
    elif black_str:
        return f'[FEN "{turn_char}:{black_str}"]'
    else:
        return f'[FEN "{turn_char}:"]'  # Brak figur na planszy, choć mało prawdopodobne


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

    # Obsługa promocji po wykonaniu ruchu
    r_last, c_last = field_to_coords(squares[-1])
    piece_at_dest = board[r_last][c_last]
    if piece_at_dest == 'w' and r_last == 0:  # Biały pionek dociera do pierwszego rzędu przeciwnika
        board[r_last][c_last] = 'W'  # Promuj na białego króla
    elif piece_at_dest == 'b' and r_last == 9:  # Czarny pionek dociera do pierwszego rzędu przeciwnika
        board[r_last][c_last] = 'B'  # Promuj na czarnego króla


def parse_game(moves_str):
    import re
    moves = re.findall(r"\d+[\-x]\d+(?:[\-x]\d+)*", moves_str)
    board = initial_board()
    turn = 'w'
    positions = []

    # Dodaj początkową pozycję przed jakimikolwiek ruchami
    positions.append(to_pdn_fen(board, turn))

    # for move in moves:
    #     apply_move(board, move)
    #     turn = 'b' if turn == 'w' else 'w'
    #     fen = to_pdn_fen(board, turn)
    #     positions.append(fen)
    # return positions
    for i, move in enumerate(moves):
        try:
            from_field = int(re.match(r'\d+', move).group())
            to_field = int(re.findall(r'\d+', move)[-1])
            if not (1 <= from_field <= 50) or not (1 <= to_field <= 50):
                # 0-2 I 2-0 TO WYNIK PARTII, WIĘC W TYM MIEJSCU KOŃCZYMY ANALIZĘ
                return positions
        except Exception as e:
            print(f"⚠️ Błąd przy analizie ruchu {move}: {e}")
            return positions

        apply_move(board, move)
        turn = 'b' if turn == 'w' else 'w'
        fen = to_pdn_fen(board, turn)
        positions.append(fen)

    return positions


def convert_moves():
    input_file = "moves_001.txt"
    output_file = "converted_moves.txt"
    with open(input_file, "r") as f:
        games = f.readlines()

    with open(output_file, "w", encoding="utf-8") as out:
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
                out.write(f"Game {idx + 1} — BŁĄD: {e}\n\n")


# === GŁÓWNY PROGRAM ===
def main():
    print(f"Pobieranie gier dla użytkownika: {USERNAME}")
    download_games()

    print("\nEkstrakcja ruchów...")
    parse_pdn_file()

    print("\nKonwersja do PDN FEN...")
    with open("moves_001.txt", "r") as f:
        print(f"Liczba partii do przetworzenia: {len(f.readlines())}")

    convert_moves()

    print("\nGotowe! Wynik zapisano w converted_moves.txt")


if __name__ == "__main__":
    main()