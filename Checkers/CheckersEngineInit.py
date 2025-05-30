from draughts import Board, Move, WHITE, BLACK
from draughts.engine import HubEngine, Limit
import os
# inicjalizacja silnika
def main():
    engine = None
    try:
        engine = HubEngine([
            r"C:\sem4\SI\project\ProjectSI\Checkers\scan_31\scan.exe",
            "hub"
        ])
        engine.hub()
        engine.init()
        limit = Limit(time=10)
        board = Board("startpose")
        move_data = engine.play(board, limit, ponder=True)
        best_move: Move = move_data.move
        print("Najlepszy ruch: ", best_move.pdn_move)

        path = r"C:\sem4\SI\project\ProjectSI\Checkers\scan_31\scan.exe"
        print(f"Czy plik istnieje: {os.path.exists(path)}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
    finally:
        if engine:
            # Upewnij się, że silnik otrzyma komendę 'quit'
            engine.quit()
            print("Silnik Scan został zamknięty.")


if __name__ == '__main__':
    main()