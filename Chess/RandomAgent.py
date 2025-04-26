import random
# Randomowy agent wybierający losowy ruch
class RandomAgent:
    def __init__(self):
        pass

    def select_move(self, legal_moves):
        if legal_moves:
            return random.choice(legal_moves)
        else:
            return None