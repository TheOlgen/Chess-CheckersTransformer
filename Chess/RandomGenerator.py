from time import sleep

import chess
#import python-chess as chess
import Evaluator
import RandomAgent

# Inicjalizacja
board = chess.Board()
agent = RandomAgent.RandomAgent()
evaluator = Evaluator.Evaluator()

# Gra
while not board.is_game_over():
    print(board)
    print("Ocena pozycji:", evaluator.evaluate(board.fen()))

    #print(list(board.legal_moves))
    move = agent.select_move(list(board.legal_moves))
    if move is None:
        break  # brak ruchów (koniec gry)

    print(f"Agent wykonuje ruch: {move}")
    board.push(move)
    sleep(1)

print(board)
print("Gra zakończona:", board.result())
