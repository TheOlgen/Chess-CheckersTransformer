import torch
from torch.utils.data import IterableDataset
from Checkers.Database.SQL_checkers import get_positions

class CheckersStreamDataset(IterableDataset):
    def __init__(self, chunk_size: int = 1000):
        # chunk_size: ile rekordów pobieramy na raz z bazy
        self.chunk_size = chunk_size

    def __iter__(self):
        while True:
            batch = get_positions(self.chunk_size)
            if not batch:
                break

            for board, best_move in batch:
                board_tensor = torch.Tensor(board)
                yield board_tensor, torch.tensor(best_move, dtype=torch.long)

