import csv

import chess
#import python-chess as chess
import Evaluator as eval
from chess.Database.SQL_chess import init_db, add_position, show_database

# Inicjalizacja
evaluator = eval.Evaluator()
file = open("bestMoveData.txt", 'a')
init_db()

with open('chessData.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    count = 0
    for row in reader:
        fen = row[0]
        best_move = evaluator.evaluate(fen)
        #file.write(fen + " ; " + best_move + "\n") #debug
        add_position(fen, best_move)
        break


file.close()
show_database()

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


