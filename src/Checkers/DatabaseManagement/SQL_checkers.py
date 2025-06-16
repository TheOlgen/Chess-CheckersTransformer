import sqlite3
import csv
import os
import re # Added import for regex

# Ścieżka do pliku CSV (wraz z nazwą pliku):
CSV_PATH = 'evaluation_001.csv'


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'draughts_positions.db')


def init_db():
    """Inicjalizacja bazy danych z unikalnymi pozycjami"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdn TEXT NOT NULL UNIQUE,
        current_player TEXT NOT NULL DEFAULT 'W',
        terminated BOOLEAN NOT NULL DEFAULT FALSE,
        best_move TEXT NOT NULL DEFAULT 'unknown',
        added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()
    print(f"Zainicjalizowano bazę danych w: {DB_PATH}")


def import_pdn_from_csv():
    """Importuje pozycje z CSV do bazy, pomijając duplikaty"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    imported = 0
    duplicates = 0

    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as file:  # Added encoding for safety
            reader = csv.reader(file)
            # Skip header row if it exists (assuming the first row is a header)
            header = next(reader, None)
            if header and header[0] == "FEN":  # Check if it looks like a header
                print(f"Pominięto nagłówek CSV: {header}")

            for row_num, row in enumerate(reader, 1):
                if not row or not row[0].strip():
                    continue

                # The FEN from your previous conversion is in row[0]
                pdn_fen = row[0].strip()
                # The best_move from your previous conversion is in row[1]
                best_move = row[1].strip() if len(row) > 1 else 'unknown'

                # Extract current_player from the FEN string for better consistency
                # Example FEN: [FEN "W:W18,24:B12,16"] -> player is 'W'
                current_player_match = re.search(r'\[FEN "(W|B):', pdn_fen)
                current_player = current_player_match.group(1) if current_player_match else 'W'  # Default to White

                try:
                    c.execute(
                        "INSERT OR IGNORE INTO positions (pdn, current_player, best_move) VALUES (?, ?, ?)",
                        (pdn_fen, current_player, best_move)
                    )
                    if c.rowcount == 1:
                        imported += 1
                    else:
                        duplicates += 1
                except sqlite3.Error as se:  # Catch specific SQLite errors
                    print(f"Błąd SQLite w wierszu {row_num} ({pdn_fen}): {se}")
                    continue
                except Exception as e:
                    print(f"Ogólny błąd w wierszu {row_num} ({pdn_fen}): {e}")
                    continue

        conn.commit()
        print(f"Zakończono import:\n- Nowe pozycje: {imported}\n- Pominięte duplikaty: {duplicates}")

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku CSV w ścieżce: {CSV_PATH}")
    except Exception as e:
        print(f"Krytyczny błąd podczas importu z CSV: {e}")
    finally:
        conn.close()


def show_database(limit=20):
    """Wyświetla zawartość bazy danych"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # Pobierz liczbę wszystkich rekordów
        c.execute("SELECT COUNT(*) FROM positions")
        total = c.fetchone()[0]

        # Pobierz przykładowe rekordy
        c.execute(f"SELECT * FROM positions ORDER BY added DESC LIMIT {limit}")
        rows = c.fetchall()

        print(f"\nBaza danych: {DB_PATH}")
        print(f"Łączna liczba pozycji: {total}\n")

        print("{:<5} {:<10} {:<60} {:<15} {:<10} {:<20}".format(
            "ID", "Gracz", "PDN (FEN)", "Najlepszy ruch", "Sprawdzono", "Data dodania"
        ))
        print("-" * 125)

        for row in rows:
            # pdn (FEN) jest w row[1], current_player w row[2], terminated w row[3], best_move w row[4], added w row[5]
            pdn_display = (row[1][:57] + '...') if len(row[1]) > 60 else row[1]
            terminated_display = "TAK" if row[3] else "NIE"
            print("{:<5} {:<10} {:<60} {:<15} {:<10} {:<20}".format(
                row[0], row[2], pdn_display, row[4], terminated_display, row[5]
            ))

        if total > limit:
            print(f"\nWyświetlono {limit} z {total} rekordów. Pełna baza zawiera więcej pozycji.")

    except Exception as e:
        print(f"Błąd podczas wyświetlania bazy danych: {e}")
    finally:
        conn.close()


def get_positions(chunk_size=200, offset=0):
    rows = []
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(f"SELECT pdn, best_move FROM positions LIMIT ? OFFSET ?", (chunk_size, offset))
        rows = c.fetchall()
    except Exception as e:
        print(f"Błąd podczas pobierania pozycji z bazy danych w chunkach: {e}")
    finally:
        conn.close()
    return rows

def restart_terminated():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    update_query = f"UPDATE positions SET terminated = 0"
    c.execute(update_query)
    conn.commit()


if __name__ == "__main__":
    print("=== Warcabowy menedżer baz danych ===")
    print(f"Ścieżka do CSV: {CSV_PATH}")
    print(f"Ścieżka do bazy: {DB_PATH}\n")

    init_db()

    # Importuj dane tylko, jeśli plik CSV istnieje
    if os.path.exists(CSV_PATH):
        import_pdn_from_csv()
    else:
        print(f"Pominięto import z CSV: plik '{CSV_PATH}' nie istnieje. Upewnij się, że wygenerowałeś go wcześniej.")

    #restart_terminated()
    show_database()
