from draughts import Board, Move, WHITE, BLACK
from draughts.engine import HubEngine, Limit

engine = HubEngine([r"C:\ScanEngine\scan_31\scan.exe", "hub"], cwd=r"C:\ScanEngine\scan_31") #ścieżki mogą nie zadziałać pewnie potrzebna zmiana

try:
    engine.init()
    limit = Limit(1000)
    board = Board(variant="standard", fen="startpos")
    move_data = engine.play(board, limit, ponder=True)
    best_move: Move = move_data.move
    print("Najlepszy ruch: ", best_move.pdn_move)
except Exception as e:
    print(e)
finally:
    engine.quit()
    engine.kill_process()

