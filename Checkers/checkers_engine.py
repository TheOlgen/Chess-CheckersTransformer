import copy
import re
import csv

def parse_simple_pdn(pdn_str):
    rows_str, turn = pdn_str.strip().split()
    rows = rows_str.strip().split("/")

    board = []
    for row in rows:
        parsed_row = []
        tokens = re.findall(r'\d+|[wbWB]', row)
        for token in tokens:
            if token.isdigit():
                parsed_row.extend(['.'] * int(token))
            else:
                parsed_row.append(token)
        board.append(parsed_row)

    return board, turn

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
                # bicie
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

# === Zapis do pliku CSV ===
if __name__ == "__main__":
    input_filename = "converted_moves.txt"
    output_filename = "evaluation_001.csv"

    with open(input_filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(output_filename, mode="w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["FEN", "najlepszy_ruch"])

        for line in lines:
            try:
                board, player = parse_simple_pdn(line)
                move = best_move(board, player)
                if move is not None:
                    from_sq = position_to_square(*move[0])
                    to_sq = position_to_square(*move[1])
                    best_move_str = f"{from_sq}-{to_sq}"
                    writer.writerow([line, best_move_str])               
            except Exception as e:
                # Nie zapisujemy linii jeśli jest błąd
                # Możesz też tu dodać print lub logging, jeśli chcesz wiedzieć o błędach
                continue
