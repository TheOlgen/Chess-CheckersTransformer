import re

def extract_moves(pdn_moves_text):
    # Usuwamy komentarze w {}, np. {[%clock ...]}
    no_comments = re.sub(r'\{[^}]*\}', '', pdn_moves_text)

    # Usuwamy numery ruchów, np. "1." lub "1..."
    no_move_numbers = re.sub(r'\d+\.+', '', no_comments)

    # Podzielmy na tokeny
    tokens = no_move_numbers.split()

    # Filtrujemy tokeny, które wyglądają jak ruchy, np. "33-28", "34x23"
    moves = [t for t in tokens if re.match(r'^\d+[-x]\d+$', t)]

    return ' '.join(moves)

def parse_pdn_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Partie są oddzielone podwójnym enterem
    games = content.strip().split('\n\n')

    moves_only_list = []

    for game in games:
        # Usuwamy nagłówek (linie zaczynające się od '[')
        # i pozostawiamy tylko ruchy (ostatnia część po nagłówkach)
        lines = game.splitlines()
        moves_lines = [line for line in lines if not line.startswith('[')]
        moves_text = ' '.join(moves_lines).strip()

        # Jeśli ruchy istnieją
        if moves_text:
            moves_only = extract_moves(moves_text)
            if moves_only:  # jeśli nie puste
                moves_only_list.append(moves_only)

    # Zapis do pliku wynikowego: każda partia w osobnej linii
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in moves_only_list:
            f_out.write(line + '\n')

def my_main():
    input_pdn = 'lidraughts_games_001.pdn'        # tutaj podaj nazwę pliku wejściowego
    output_moves = 'moves_001.txt'      # plik wyjściowy z samymi ruchami

    parse_pdn_file(input_pdn, output_moves)
    print(f"Gotowe! Ruchy zapisane w {output_moves}")
