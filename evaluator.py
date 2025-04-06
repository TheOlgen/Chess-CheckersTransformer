from typing import Tuple
import numpy as np
import math

from game_functions import Game


class Evaluator:
    # win_length - liczba pionków potrzebna do zwycięstwa
    def __evaluate_line(self, line, player, win_length):
        opponent = -player
        if opponent in line:
            return 0
        count = np.count_nonzero(line == player)

        if count == 0:
            return 1
        else:
            return 10 ** count

    def evaluate_board(self, board, win_length, player):

        n = board.shape[0]
        score = 0

        def evaluate_sequences(arr):
            seq_score = 0
            for i in range(len(arr) - win_length + 1):
                segment = arr[i:i + win_length]
                seq_score += self.__evaluate_line(segment, player, win_length)
            return seq_score

        for i in range(n):
            score += evaluate_sequences(board[i, :])

        for j in range(n):
            score += evaluate_sequences(board[:, j])

        for k in range(-n + win_length, n - win_length + 1):
            diag = np.diag(board, k=k)
            if len(diag) >= win_length:
                score += evaluate_sequences(diag)

        flipped = np.fliplr(board)
        for k in range(-n + win_length, n - win_length + 1):
            diag = np.diag(flipped, k=k)
            if len(diag) >= win_length:
                score += evaluate_sequences(diag)

        return score

    def minimax(self, board, depth, current, side):
        result = Game.check_winner(board)
        if result is not None:
            if result == side:
                return 10 - depth, None
            elif result == 0:
                return 0, None
            else:
                return depth - 10, None

        if current == side:
            best_score = -np.inf
            best_move = None
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = current
                        score, _ = self.minimax(board, depth + 1, -current, side)
                        board[i, j] = 0
                        if score > best_score:
                            best_score = score
                            best_move = (i, j)
            return best_score, best_move
        else:
            best_score = np.inf
            best_move = None
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = current
                        score, _ = self.minimax(board, depth + 1, -current, side)
                        board[i, j] = 0
                        if score < best_score:
                            best_score = score
                            best_move = (i, j)
            return best_score, best_move

    def __get_best_move(self, board, side=None):
        if side is None:
            side = Game.get_current_turn(board)
        score, move = self.minimax(board.copy(), 0, side, side)
        return move

    def get_label(self, board):
        act_turn = Game.get_current_turn(board)
        best_move = self.__get_best_move(board)
        evaluation = self.evaluate_board(board, board.shape[0], act_turn)
        return best_move, evaluation
