import sqlite3
import csv


# Ścieżka do pliku CSV (wraz z nazwą pliku)
CSV_PATH = ''  # Przykład: 'C:/warcaby/dane.csv'

# Ścieżka do bazy danych (wraz z nazwą pliku .db)
DB_PATH = ''  # Przykład: 'C:/warcaby/baza.db'




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
        with open(CSV_PATH, 'r') as file:
            reader = csv.reader(file)
            for row_num, row in enumerate(reader, 1):
                if not row or not row[0].strip():
                    continue

                pdn = row[0].strip()
                current_player = row[1].strip() if len(row) > 1 else 'W'

                try:
                    c.execute(
                        "INSERT OR IGNORE INTO positions (pdn, current_player) VALUES (?, ?)",
                        (pdn, current_player)
                    )
                    if c.rowcount == 1:
                        imported += 1
                    else:
                        duplicates += 1
                except Exception as e:
                    print(f"Błąd w wierszu {row_num}: {e}")
                    continue

        conn.commit()
        print(f"Zakończono import:\n- Nowe pozycje: {imported}\n- Pominięte duplikaty: {duplicates}")

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku CSV w ścieżce: {CSV_PATH}")
    except Exception as e:
        print(f"Krytyczny błąd: {e}")
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

        print("{:<5} {:<8} {:<40} {:<15} {:<20}".format(
            "ID", "Gracz", "PDN", "Najlepszy ruch", "Data dodania"
        ))
        print("-" * 90)

        for row in rows:
            print("{:<5} {:<8} {:<40} {:<15} {:<20}".format(
                row[0], row[2], (row[1][:37] + '...') if len(row[1]) > 40 else row[1],
                row[4], row[5]
            ))

        if total > limit:
            print(f"\nWyświetlono {limit} z {total} rekordów. Pełna baza zawiera więcej pozycji.")

    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        conn.close()



if __name__ == "__main__":
    print("=== Warcabowy menedżer baz danych ===")
    print(f"Ścieżka do CSV: {CSV_PATH}")
    print(f"Ścieżka do bazy: {DB_PATH}\n")

    init_db()
    import_pdn_from_csv()
    show_database()

