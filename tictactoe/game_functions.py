import numpy as np


class Game:
    @staticmethod
    def check_winner(board):
        for i in range(3):
            if board[i, 0] == board[i, 1] == board[i, 2] != 0:
                return board[i, 0]
        for j in range(3):
            if board[0, j] == board[1, j] == board[2, j] != 0:
                return board[0, j]

        if board[0, 0] == board[1, 1] == board[2, 2] != 0:
            return board[0, 0]
        if board[0, 2] == board[1, 1] == board[2, 0] != 0:
            return board[0, 2]

        if not np.any(board == 0):
            return 0
        return None

    @staticmethod
    def get_current_turn(board, first_player=1):
        count_first = np.count_nonzero(board == first_player)
        count_second = np.count_nonzero(board == -first_player)
        return first_player if count_first == count_second else -first_player


