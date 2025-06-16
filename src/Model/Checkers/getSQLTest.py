from SQL_checkers import get_positions

for custom_pdn_fen, best_move_str in get_positions(200):
    try:
        print(custom_pdn_fen)
    except Exception as e:
        print(f"Błąd przetwarzania pozycji: {custom_pdn_fen}, ruch: {best_move_str} - {e}")