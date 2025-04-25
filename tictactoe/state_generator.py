import numpy as np

from game_functions import Game


class StateGenerator:
    def __init__(self):
        self.size = 3

    def get_possible_moves(self, board):
        possible_moves = []
        size = board.shape[0]
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    possible_moves.append((i, j))
        return possible_moves

    def __generate_states(self, board, size, states, depth):



        if Game.check_winner(board) == 1 or Game.check_winner(board) == -1:
            return
        if not self.get_possible_moves(board):
            return

        if(depth == 0):
            return

        act_turn = Game.get_current_turn(board, 1)
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    new_board = np.copy(board)
                    new_board[i][j] = act_turn
                    states.append(new_board)
                    self.__generate_states(new_board, size, states, depth - 1)

    @staticmethod
    def save_states(states, filename="states.npy"):
        np.save(filename, np.array(states, dtype=int))

    @staticmethod
    def load_states(filename="states.npy"):
        try:
            return list(np.load(filename, allow_pickle=True).astype(int))
        except FileNotFoundError:
            return None

    def generate_w_save_game_states(self, size, depth, filename="states.npy"):
        states = []
        board = np.zeros((size, size))
        self.__generate_states(board, size, states, depth)
        StateGenerator.save_states(states, filename)
        return states


