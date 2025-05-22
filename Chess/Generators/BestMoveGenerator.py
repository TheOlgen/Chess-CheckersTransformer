import csv

import chess
#import python-chess as chess
import Evaluator as eval
#import pandas as pd

# Inicjalizacja
#board = chess.Board()
evaluator = eval.Evaluator()
file = open("bestMoveData.txt", 'a')
#TODO: PLIK Z planszami
#inputs = open("boards.txt", 'r')

with open('chessData.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    count = 0
    for row in reader:
        count += 1
        if count >= 20:
            break
        fen = row[0]
        best_move = evaluator.evaluate(fen)
        file.write(fen + " ; " + best_move + "\n")


file.close()

# with open("boards.txt", 'r') as boards:
#     while True:
#         fen = boards.readline()
#         if fen == "":
#             break
#         print(fen)
#         best_move = evaluator.evaluate(fen)
#         print("Best move:", best_move)
#         file.write(fen + " ; " + best_move + "\n")



    # Gra


    #print(list(board.legal_moves))


