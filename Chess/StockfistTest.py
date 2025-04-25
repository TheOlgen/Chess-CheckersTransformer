
from stockfish import Stockfish

sf = Stockfish(
        path="Stockfish/stockfish.exe",
        parameters={"Threads": 8, "Hash": 1024}
     )
sf.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
sf._go()

print(sf.get_best_move())         # ruch w UCI, np. e2e4
print(sf.get_evaluation())        # {'type': 'cp', 'value': 23}
