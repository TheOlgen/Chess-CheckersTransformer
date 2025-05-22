import sqlite3
import csv


def init_db(db_file='chess_positions.db'):
    """Inicjalizacja bazy danych z unikalnymi FEN"""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fen TEXT NOT NULL UNIQUE,
        terminated BOOLEAN NOT NULL DEFAULT FALSE,
        best_move TEXT NOT NULL DEFAULT 'unknown',
        added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


def import_fen_from_csv(csv_file, db_file='chess_positions.db'):
    """Importuje FEN z CSV, pomijając duplikaty"""
    init_db(db_file)  # Upewnij się, że tabela istnieje
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    imported = 0
    duplicates = 0

    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or not row[0].strip():
                    continue

                fen = row[0].strip()
                try:
                    c.execute(
                        "INSERT OR IGNORE INTO positions (fen) VALUES (?)",
                        (fen,)
                    )
                    if c.rowcount > 0:
                        imported += 1
                    else:
                        duplicates += 1
                except sqlite3.IntegrityError:
                    duplicates += 1

        conn.commit()
        print(f"Zaimportowano {imported} nowych pozycji, pominięto {duplicates} duplikatów")
    except Exception as e:
        print(f"Błąd: {e}")
        conn.rollback()
    finally:
        conn.close()
    return imported


def show_database(db_file='chess_positions.db'):
    """Wyświetla zawartość bazy danych"""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    try:
        c.execute("SELECT * FROM positions")
        rows = c.fetchall()

        print("\nZawartość bazy:")
        print("{:<5} {:<60} {:<10} {:<15} {:<20}".format(
            "ID", "FEN", "Terminated", "Best Move", "Added"
        ))
        print("-" * 120)
        for row in rows:
            print("{:<5} {:<60} {:<10} {:<15} {:<20}".format(
                row[0], row[1], str(row[2]), row[3], row[4]
            ))
    finally:
        conn.close()


if __name__ == "__main__":
    # Przykład użycia:
    init_db()
    import_fen_from_csv('data.csv')  # Upewnij się, że plik data.csv istnieje
    show_database()