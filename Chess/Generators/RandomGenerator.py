import chess
#import python-chess as chess
import Evaluator
import RandomAgent

# Inicjalizacja
board = chess.Board()
agent = RandomAgent.RandomAgent()
evaluator = Evaluator.Evaluator()
file = open("randomData.txt", 'a')

# Gra
while True:
    if board.is_game_over():
        board = chess.Board()
    print(board)
    best_move, posEval = evaluator.evaluate(board.fen())
    print("Ocena pozycji:", best_move, posEval)
    file.write(board.fen() + " " + best_move + " " + str(posEval) + "\n")

    #print(list(board.legal_moves))
    move = agent.select_move(list(board.legal_moves))
    if move is None:
        break  # brak ruchów (koniec gry)

    print(f"Agent wykonuje ruch: {move}")
    board.push(move)

print(board)
print("Gra zakończona:", board.result())
