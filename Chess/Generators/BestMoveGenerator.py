import chess
#import python-chess as chess
import Evaluator as eval

# Inicjalizacja
board = chess.Board()
evaluator = eval.Evaluator()
file = open("bestMoveData.txt", 'a')

# Gra
while True:
    if board.is_game_over():
        board = chess.Board()
    print(board)
    best_move, posEval = evaluator.evaluate(board.fen())
    print("Ocena pozycji:", best_move, posEval)
    file.write(board.fen() + " " + best_move + " " + str(posEval) + "\n")

    #print(list(board.legal_moves))
    if best_move is None:
        break  # brak ruchów (koniec gry)

    best_move = chess.Move.from_uci(best_move)

    print(f"Agent wykonuje ruch: {best_move}")
    board.push(best_move)

print(board)
print("Gra zakończona:", board.result())
