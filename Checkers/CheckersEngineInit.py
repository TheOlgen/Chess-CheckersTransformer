from draughts import Board, Move, WHITE, BLACK
from draughts.engine import HubEngine, Limit
import os
import csv
import re  # Potrzebne do ekstrakcji FEN z "Game X:" linii

# --- KONFIGURACJA ---
ENGINE_PATH = r"C:\sem4\SI\project\ProjectSI\Checkers\scan_31\scan.exe"
INPUT_DATA_FILE = r"C:\sem4\SI\project\ProjectSI\Checkers\converted_moves.txt"
OUTPUT_CSV_FILE = r"C:\sem4\SI\project\ProjectSI\Checkers\evaluation_001.csv"
ANALYSIS_TIME_LIMIT = 10  # Czas w sekundach na analizę każdej pozycji


def main():
    """
    Reads draughts positions from converted_moves.txt, analyzes each using the Scan engine,
    and saves the FEN and best move to evaluation_001.csv.
    """
    engine = None

    # --- Weryfikacja ścieżek ---
    if not os.path.exists(ENGINE_PATH):
        print(f"BŁĄD: Plik silnika nie istnieje pod ścieżką: {ENGINE_PATH}")
        print("Upewnij się, że ścieżka do 'scan.exe' jest prawidłowa.")
        return

    if not os.path.exists(INPUT_DATA_FILE):
        print(f"BŁĄD: Plik wejściowy danych '{INPUT_DATA_FILE}' nie istnieje.")
        return

    # --- Inicjalizacja silnika ---
    try:
        print(f"Ładowanie silnika z: {ENGINE_PATH}")
        engine = HubEngine([ENGINE_PATH, "hub"])
        engine.hub()
        engine.init()
        print("✅ Silnik Scan zainicjalizowany.")

        limit = Limit(time=ANALYSIS_TIME_LIMIT)

        # --- Przygotowanie do zapisu CSV ---
        # Użyj 'w' do nadpisania pliku, 'a' do dopisania (jeśli chcesz kontynuować)
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(["FEN", "najlepszy_ruch"])  # Zapis nagłówka CSV

            with open(INPUT_DATA_FILE, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            print(
                f"\nAnalizowanie {len(lines)} linii z pliku '{INPUT_DATA_FILE}' i zapisywanie do '{OUTPUT_CSV_FILE}'...")

            # Wskazówka: Plik converted_moves.txt ma format "Game X:\n[FEN "Y"]\n\n"
            # Musimy wyodrębnić tylko linie FEN.
            fen_lines = [line.strip() for line in lines if line.strip().startswith('[FEN "')]

            if not fen_lines:
                print("Brak pozycji FEN do analizy w pliku wejściowym.")
                return

            print(f"Znaleziono {len(fen_lines)} pozycji FEN do analizy.")

            for i, fen_line in enumerate(fen_lines):
                try:
                    # 'fen_line' to już cały ciąg '[FEN "..."']
                    board = Board(fen_line)
                    print(f"\n--- Analiza pozycji {i + 1}/{len(fen_lines)}: {fen_line} ---")

                    move_data = engine.play(board, limit, ponder=True)
                    best_move: Move = move_data.move

                    if best_move:
                        best_move_str = best_move.pdn_move
                        print(f"Najlepszy ruch: {best_move_str}")
                    else:
                        best_move_str = "Brak ruchów"  # lub inna wartość, jeśli silnik nie znalazł ruchu
                        print("Brak sugerowanych ruchów dla tej pozycji.")

                    # Zapisz FEN i najlepszy ruch do pliku CSV
                    csv_writer.writerow([fen_line, best_move_str])

                except Exception as e_pos:
                    print(f"❌ Błąd podczas analizy pozycji '{fen_line}': {e_pos}")
                    # Nadal zapisz wiersz do CSV, aby nie utracić rekordu, ale oznacz błąd
                    csv_writer.writerow([fen_line, f"BŁĄD: {e_pos}"])
                    continue  # Przejdź do następnej pozycji

        print(f"\n✅ Zakończono analizę i zapisano wyniki do '{OUTPUT_CSV_FILE}'")

    except Exception as e_main:
        print(f"❌ Wystąpił ogólny błąd silnika lub zapisu pliku: {e_main}")
    finally:
        if engine:
            try:
                engine.quit()
                print("\nSilnik Scan został zamknięty.")
            except Exception as e_quit:
                print(f"Ostrzeżenie: Błąd podczas zamykania silnika: {e_quit}")


if __name__ == '__main__':
    main()