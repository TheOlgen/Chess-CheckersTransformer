from stockfish import Stockfish

class Evaluator:
    def __init__(self):
        self.sf = Stockfish(
            path="../Stockfish/stockfish.exe",
            parameters={"Threads": 8, "Hash": 1024}
        )

    def evaluate(self, fen_position):
        self.sf.set_fen_position(fen_position)
        return self.sf.get_best_move()
