import numpy as np
import pickle
import os

from evaluator import Evaluator
from state_generator import StateGenerator


class StatesLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def save_processed_states(labeled_states, filename="labeled_states.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(labeled_states, f)

    @staticmethod
    def load_processed_states(filename="labeled_states.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        return []

    def label_states(self, states_file_name, output_file_name, max_states=None):
        evaluator = Evaluator()
        states = StateGenerator.load_states(states_file_name)

        labeled_states = StatesLoader.load_processed_states(output_file_name)
        existing_boards = {tuple(state["board"].flatten()) for state in labeled_states}

        if max_states is None:
            max_states = len(states)

        for idx, board in enumerate(states[:max_states]):
            board_tuple = tuple(board.flatten())

            #if board_tuple in existing_boards:
                #continue

            best_move, evaluation = evaluator.get_label(board)

            labeled_states.append({
                "board": board,
                "best_move": best_move,
                "score": evaluation
            })

            if idx % self.batch_size == 0:
                StatesLoader.save_processed_states(labeled_states, output_file_name)
                print(f"Zapisano {idx}/{len(states)} stanów...")

        StatesLoader.save_processed_states(labeled_states, output_file_name)
        print("Przetwarzanie zakończone!")
        return labeled_states
