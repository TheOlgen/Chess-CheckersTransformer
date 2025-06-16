from typing import List, Tuple
import torch
from torch.utils.data import IterableDataset
import re
from CheckersTraining.CheckersUtils import move_to_index, fen_to_board_tensor, custom_pdn_fen_to_standard_fen
from Checkers.DatabaseManagement.SQL_checkers import get_positions

class GameDataset(IterableDataset):
    def __init__(self, data_preprocessing, chunk_size: int = 200, records_per_epoch: int = None):
        self.chunk_size = chunk_size
        self.records_per_epoch = records_per_epoch
        self.data_preprocessing = data_preprocessing

        self.current_epoch = 0
        self.db_offset = 0

    def __iter__(self):
        # get_positions now yields (pdn_fen_string, best_move_string)

        if self.records_per_epoch is not None:
            start = self.current_epoch * self.records_per_epoch + self.db_offset
            fetched = 0

            while fetched < (self.records_per_epoch - self.db_offset):
                batch = get_positions(limit=self.chunk_size, offset=start)
                if not batch:
                    break
                for custom_pdn_fen, best_move_str in batch:
                    try:
                        board_tensor, move_idx = self.data_preprocessing(cus)
                        # Convert custom PDN FEN to standard FEN first
                        standard_fen = custom_pdn_fen_to_standard_fen(custom_pdn_fen)
                        board_tensor = fen_to_board_tensor(standard_fen)
                        move_idx = move_to_index(best_move_str)

                        yield board_tensor, torch.tensor(move_idx, dtype=torch.long)
                    except Exception as e:
                        print(f"Błąd przetwarzania pozycji: {custom_pdn_fen}, ruch: {best_move_str} - {e}")



    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "db_offset": self.db_offset
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]
        self.db_offset = state_dict["db_offset"]