from evaluator import Evaluator
import numpy as np

from state_generator import StateGenerator
from states_loader import StatesLoader

evaluator = Evaluator()
board1 = np.array([
    [1, 0, -1],
    [0, 1, 0],
    [0, -1, 0]
])

board2 = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

#best_move, evaluation = evaluator.get_label(board1)
#print(best_move)
#print(evaluation)


state_generator = StateGenerator()
#states = state_generator.generate_w_save_game_states(3)

#states = StateGenerator.load_states("states.npy")
#print(states)

#states_loader = StatesLoader(100)
#states_with_labels_3x3 = states_loader.label_states("states.npy", "states_with_labels_3x3.pkl")
#states_with_labels_3x3 = StatesLoader.load_processed_states("states_with_labels_3x3.pkl")
#print(states_with_labels_3x3[:15])

states = state_generator.generate_w_save_game_states(4, 2,"states_4x4_1")
print(states)