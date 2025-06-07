import os
import csv
import random
import time
from typing import List, Tuple, Optional
from draughts import Board, Move, WHITE, BLACK
from draughts.engine import HubEngine, Limit
from Model.Checkers.CheckersDataset import custom_pdn_fen_to_standard_fen, board_tensor_to_fen, fen_to_board_tensor
import torch


class CheckersSelfPlayGenerator:
    def __init__(self, engine_path: str, output_csv: str = "self_play_data.csv",
                 time_limit: int = 5):
        """
        Inicjalizuje generator danych self-play dla warcabów.

        Args:
            engine_path: Ścieżka do silnika scan.exe
            output_csv: Plik wyjściowy CSV z danymi treningowymi
            time_limit: Limit czasu na ruch w sekundach
        """
        self.engine_path = engine_path
        self.output_csv = output_csv
        self.time_limit = time_limit
        self.engine = None
        self.limit = Limit(time=time_limit)

    def initialize_engine(self) -> bool:
        """Inicjalizuje silnik warcabowy."""
        try:
            if not os.path.exists(self.engine_path):
                print(f"❌ Silnik nie istnieje: {self.engine_path}")
                return False

            print(f"🔧 Inicjalizacja silnika: {self.engine_path}")
            self.engine = HubEngine([self.engine_path, "hub"])
            self.engine.hub()
            self.engine.init()
            print("✅ Silnik zainicjalizowany")
            return True
        except Exception as e:
            print(f"❌ Błąd inicjalizacji silnika: {e}")
            return False

    def create_random_opening_position(self) -> Board:
        """
        Tworzy losową pozycję warcabową przez wykonanie kilku losowych ruchów
        z pozycji początkowej.
        """
        board = Board()  # Pozycja początkowa

        # Wykonaj 3-8 losowych ruchów, aby uzyskać różnorodne pozycje
        num_moves = random.randint(3, 8)

        for _ in range(num_moves):
            legal_moves = list(board.legal_moves())
            if not legal_moves:
                break

            # Wybierz losowy legalny ruch
            random_move = random.choice(legal_moves)
            board.push(random_move)

            # Sprawdź, czy gra się nie skończyła
            if board.is_over():
                # Jeśli gra się skończyła, cofnij ostatni ruch i przerwij
                board.pop()
                break

        return board

    def create_mid_game_position(self) -> Board:
        """
        Tworzy pozycję z środkowej fazy gry przez symulację dłuższej partii.
        """
        board = Board()

        # Wykonaj 10-25 ruchów dla pozycji środkowej gry
        num_moves = random.randint(10, 25)

        for _ in range(num_moves):
            legal_moves = list(board.legal_moves())
            if not legal_moves or board.is_over():
                break

            # Preferuj ruchy bicia (bardziej dynamiczne pozycje)
            capture_moves = [move for move in legal_moves if len(move.captures) > 0]

            if capture_moves and random.random() < 0.7:  # 70% szans na wybór bicia
                chosen_move = random.choice(capture_moves)
            else:
                chosen_move = random.choice(legal_moves)

            board.push(chosen_move)

        return board

    def create_tactical_position(self) -> Optional[Board]:
        """
        Próbuje stworzyć pozycję taktyczną z możliwościami bicia.
        """
        for _ in range(10):  # Maksymalnie 10 prób
            board = self.create_mid_game_position()

            legal_moves = list(board.legal_moves())
            capture_moves = [move for move in legal_moves if len(move.captures) > 0]

            # Jeśli istnieją bicia, to prawdopodobnie ciekawa pozycja taktyczna
            if len(capture_moves) >= 2:
                return board

        return None  # Nie udało się stworzyć pozycji taktycznej

    def generate_position_data(self, board: Board) -> Tuple[str, str]:
        try:
            fen_string = board.fen
            move_data = self.engine.play(board, self.limit, ponder=False)
            best_move = move_data.move if move_data else None

            if best_move:
                return fen_string, best_move.pdn_move
            else:
                print("⚠️ Brak najlepszego ruchu od silnika.")
                return fen_string, "Brak ruchów"

        except Exception as e:
            print(f"❌ Błąd generowania danych: {e}")
            return None, None

    #from draughts.engine import SimpleEngine, Limit

    def self_play_game(self, max_moves: int = 100):
        board = Board()
        game_data = []
        move_count = 0

        print("🎮 Start self-play")

        while not board.is_over() and move_count < max_moves:
            fen_string = board.fen
            print(f"♟️ Ruch {move_count}, FEN: {fen_string}")

            try:
                legal_moves = list(board.legal_moves())
                if not legal_moves:
                    print("⛔ Brak legalnych ruchów!")
                    break

                move_data = self.engine.play(board, Limit(time=0.1), ponder=False)
                best_move = move_data.move

                if best_move is None:
                    print("⚠️ Brak ruchu z silnika")
                    break

                best_move_str = best_move.pdn_move
                game_data.append((fen_string, best_move_str))

                board.push(best_move)
                move_count += 1

            except Exception as e:
                print(f"❌ Błąd przy ruchu {move_count}: {e}")
                break

        print(f"🏁 Zakończono po {move_count} ruchach.")
        return game_data

    def generate_random_positions(self, num_positions: int) -> List[Tuple[str, str]]:
        """
        Generuje określoną liczbę losowych pozycji z ich najlepszymi ruchami.
        """
        positions_data = []

        print(f"🎲 Generowanie {num_positions} losowych pozycji...")

        for i in range(num_positions):
            print(f"🎲 Generowanie {i} pozycji...")
            try:
                # Wybierz typ pozycji losowo
                position_type = random.choices(
                    ['opening', 'midgame', 'tactical'],
                    weights=[0.3, 0.5, 0.2]  # 30% opening, 50% midgame, 20% tactical
                )[0]

                if position_type == 'opening':
                    board = self.create_random_opening_position()
                elif position_type == 'midgame':
                    board = self.create_mid_game_position()
                else:  # tactical
                    board = self.create_tactical_position()
                    if board is None:  # Fallback do midgame
                        board = self.create_mid_game_position()

                # Sprawdź, czy pozycja nie jest zakończona
                if board.is_over():
                    continue

                fen_string, best_move_str = self.generate_position_data(board)

                if fen_string and best_move_str and best_move_str != "Brak ruchów":
                    positions_data.append((fen_string, best_move_str))

                    if (i + 1) % 50 == 0:
                        print(f"📈 Wygenerowano {i + 1}/{num_positions} pozycji")

            except Exception as e:
                print(f"❌ Błąd generowania pozycji {i + 1}: {e}")
                continue

        return positions_data

    def save_to_csv(self, data: List[Tuple[str, str]], append: bool = False):
        """Zapisuje dane do pliku CSV."""


        with open(self.output_csv, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)



            writer.writerows(data)

        print(f"💾 Zapisano {len(data)} pozycji do {self.output_csv}")

    def generate_training_data(self,
                               num_random_positions: int = 200,
                               num_self_play_games: int = 10,
                               save_batch_size: int = 100):
        """
        Główna metoda generująca dane treningowe.

        Args:
            num_random_positions: Liczba losowych pozycji do wygenerowania
            num_self_play_games: Liczba partii self-play
            save_batch_size: Co ile pozycji zapisywać do pliku
        """
        if not self.initialize_engine():
            return

        all_data = []

        try:
            # 1. Generuj losowe pozycje
            print("\n" + "=" * 50)
            print("🎯 FAZA 1: Generowanie losowych pozycji")
            print("=" * 50)

            random_data = self.generate_random_positions(num_random_positions)
            all_data.extend(random_data)

            # 2. Przeprowadź partie self-play
            print("\n" + "=" * 50)
            print("🎯 FAZA 2: Partie self-play")
            print("=" * 50)

            for game_num in range(num_self_play_games):
                print(f"\n🎮 Partia {game_num + 1}/{num_self_play_games}")
                game_data = self.self_play_game()
                all_data.extend(game_data)

                # Zapisuj co kilka partii
                if len(all_data) >= save_batch_size:
                    self.save_to_csv(all_data, append=game_num > 0)
                    all_data = []

            # Zapisz pozostałe dane
            if all_data:
                self.save_to_csv(all_data, append=True)

            print("\n" + "=" * 50)
            print("✅ GENEROWANIE DANYCH ZAKOŃCZONE")
            print("=" * 50)

        except KeyboardInterrupt:
            print("\n⚠️ Przerwano przez użytkownika")
            if all_data:
                self.save_to_csv(all_data, append=True)
                print("💾 Zapisano dotychczasowe dane")

        finally:
            if self.engine:
                try:
                    self.engine.quit()
                    print("🔧 Silnik zamknięty")
                except:
                    pass


def main():
    """Główna funkcja demonstracyjna."""
    # Konfiguracja
    ENGINE_PATH = "/Checkers/scan_31/scan.exe"
    OUTPUT_CSV = "self_play_training_data.csv"

    # Utwórz generator
    generator = CheckersSelfPlayGenerator(
        engine_path=ENGINE_PATH,
        output_csv=OUTPUT_CSV,
        time_limit=3  # 3 sekundy na ruch - szybsze generowanie
    )

    # Generuj dane treningowe
    generator.generate_training_data(
        num_random_positions=0,  # 300 losowych pozycji
        num_self_play_games=50000,  # 5 partii self-play
        save_batch_size=50  # Zapisuj co 50 pozycji
    )


if __name__ == "__main__":
    main()