import csv
import Evaluator as eval
from chess.Database.SQL_chess import init_db, add_position, show_database

# Inicjalizacja
evaluator = eval.Evaluator()
file = open("bestMoveData.txt", 'a')
init_db()

with open('chessData.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        fen = row[0]
        best_move = evaluator.evaluate(fen)
        #file.write(fen + " ; " + best_move + "\n") #debug
        add_position(fen, best_move)
        #to zakomentujcie jeśli chcecie coś dodawać do bazy
        break


file.close()
show_database()




