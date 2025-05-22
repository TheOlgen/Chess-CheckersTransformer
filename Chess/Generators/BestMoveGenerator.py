import csv

import chess
#import python-chess as chess
import Evaluator as eval

# Inicjalizacja
#board = chess.Board()
evaluator = eval.Evaluator()
file = open("bestMoveData.txt", 'a')
#TODO: PLIK Z planszami
inputs = open("boards.txt", 'r')

with open("boards.txt", 'r') as boards:
    while True:
        fen = boards.readline()
        if fen == "":
            break
        print(fen)
        best_move = evaluator.evaluate(fen)
        print("Best move:", best_move)
        file.write(fen + " ; " + best_move + "\n")



    # Gra


    #print(list(board.legal_moves))


