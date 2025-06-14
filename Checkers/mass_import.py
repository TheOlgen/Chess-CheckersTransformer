import requests

def my_main(my_username):
    # === KONFIGURACJA ===
    TOKEN = 'fGV1t6jvtfu118uO'
    USERNAME = my_username  # np. 'janek123'
    LIMIT = 100  # max 300

    # === ŻĄDANIE ===
    headers = {
        'Authorization': f'Bearer {TOKEN}',
        'Accept': 'application/x-chess-pgn'
    }

    url = f'https://lidraughts.org/api/games/user/{USERNAME}?max={LIMIT}&moves=true&tags=true&clocks=true'

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open('lidraughts_games_001.pdn', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f'✅ Zapisano {LIMIT} gier do "lidraughts_games.pdn"')
    else:
        print(f'❌ Błąd ({response.status_code}): {response.text}')