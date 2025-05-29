import torch
from BetterModel import ChessTransformer
from ChessDataset import ChessDataset


def fen_to_board(fen: str) -> torch.Tensor:
    piece_map = {
        'K': 6, 'Q': 5, 'R': 4, 'B': 3, 'N': 2, 'P': 1,
        'k': 12,'q':11,'r':10,'b':9,'n':8,'p':7
    }
    rows = fen.split(' ')[0].split('/')
    board = []
    for row in rows:
        for ch in row:
            if ch.isdigit():
                board.extend([0] * int(ch))
            else:
                board.append(piece_map.get(ch, 0))
    return torch.tensor(board, dtype=torch.long)

def main():

    test_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/1b2P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    board_tensor = fen_to_board(test_fen).unsqueeze(0)  # shape (1,64)

    model = ChessTransformer(
        d_model=512,
        max_len=64,
        num_moves=4096,
        num_heads=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        lr=3e-4
    )
    model.eval()

    # --- opcjonalnie: wczytaj checkpoint ---
    # checkpoint_path = "checkpoints/Chess-transformer-epoch=05-val/loss=1.2345.ckpt"
    # model = ChessTransformer.load_from_checkpoint(checkpoint_path)

    with torch.no_grad():
        move = model.predict_move(board_tensor.squeeze(0))
    print(f"Predicted move: \n→ {move}")

    assert isinstance(move, str), "predict_move nie zwróciło stringa!"
    assert len(move) in (4,5), "Nieoczekiwany format ruchu!"
    print("Test passed 🎉")

if __name__ == "__main__":
    main()
