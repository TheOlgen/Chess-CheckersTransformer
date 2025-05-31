from SQL_checkers import get_positions

for custom_pdn_fen, best_move_str,terminated in get_positions(200):
    try:
        print(terminated)
    except Exception as e:
        print(f"Błąd przetwarzania pozycji: {terminated}, ruch: {best_move_str} - {e}")