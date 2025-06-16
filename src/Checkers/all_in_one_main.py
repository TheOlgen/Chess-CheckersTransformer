import mass_import
import extract_moves
import checkers_converter
import checkers_engine
import os


def run_pipeline_for_user(username):
    """Uruchamia cały pipeline dla pojedynczego użytkownika"""
    print(f"\n=== Przetwarzanie użytkownika: {username} ===")

    # 1. Pobieranie gier (mass_import) - zawsze nadpisuj
    mass_import.my_main(username)

    # 2. Ekstrakcja ruchów (extract_moves) - zawsze nadpisuj
    extract_moves.my_main()

    # 3. Konwersja do FEN (checkers_converter) - zawsze nadpisuj
    checkers_converter.my_main()

    # 4. Analiza ruchów (checkers_engine) - NIE nadpisuj jeśli istnieje
    checkers_engine.my_main()

def main():
    # Sprawdź czy plik z nazwami użytkowników istnieje
    if not os.path.exists("username.txt"):
        print("Błąd: Brak pliku username.txt")
        return

    # Wczytaj wszystkich użytkowników
    with open("username.txt", "r") as f:
        usernames = [line.strip() for line in f if line.strip()]

    if not usernames:
        print("Błąd: Plik username.txt jest pusty")
        return

    # Dla każdego użytkownika wykonaj pełen pipeline
    for username in usernames:
        run_pipeline_for_user(username)


if __name__ == "__main__":
    main()