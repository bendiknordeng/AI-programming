import numpy as np


class Board:
    def __init__(self, state, player=1):
        self.state = state
        self.player = player

    def player1_won(self):
        if self.is_game_over():
            return self.player == 1 # will return True if player 1 moved to final state

    def get_state(self):
        if type(self.state) == int:
            state = self.state
        else:
            state = tuple(self.state)
        return (self.player, state)

    def __repr__(self):
        return "Player: {}, State: {}".format(self.player, self.state)

class NIMBoard(Board):
    def __init__(self, state, K, player=1):
        super().__init__(state, player)
        self.K = K

    def is_game_over(self):
        return self.state <= self.K

    def move(self, action):
        self.state -= action
        self.player = 3 - self.player #switch player after each move

    def get_legal_actions(self):
        return list(range(1, min(self.state, self.K) + 1))

    @staticmethod
    def print_move(action, board):
        player, state = board.get_state()
        remaining = "Remaining stones = {:<2}".format(state-action if state>0 else 0)
        stones = "{:<1} stones".format(action) if action > 1 else "{:<2} stone".format(action)
        return "Player {} selects {:>8}: {:>21}\n".format(player, stones, remaining)


class LedgeBoard(Board):
    def __init__(self, state, player=1):
        super().__init__(state, player)

    def is_game_over(self):
        return self.state[0] == 2

    def move(self, action):
        new_board = np.copy(self.state)
        if action == 0:
            assert new_board[0] != 0, 'There is no coin on the ledge'
            new_board[0] = 0
        else:
            i, j = action
            assert new_board[i] != 0, 'There is no coin in spot {}'.format(i)
            assert new_board[j] == 0, 'You cannot put a coin in spot {}'.format(
                j)
            new_board[j] = new_board[i]
            new_board[i] = 0
        self.state = new_board
        self.player = 3 - self.player

    def get_legal_actions(self):
        if self.state[0] == 2: return [0] # make it only possible to pick up gold if possible
        valid = []
        board = self.state
        board_length = len(self.state)
        for i in range(board_length - 1):
            if i == 0 and board[0] != 0:
                valid.append(0)
                continue
            to = []
            if board[i + 1] != 0:
                j = i
                while j >= 0 and board[j] == 0:
                    to.append(j)
                    j -= 1
            [valid.append((i + 1, j)) for j in to]
        return valid

    @staticmethod
    def print_move(action, board):
        player, state = board.get_state()
        new_board = np.copy(state)
        if action == 0:
            coin = "copper" if state[0] == 1 else "gold"
            new_board[0] = 0
            return "P{} picks up {}: {}\n".format(player, coin, str(new_board))
        else:
            coin = "copper" if state[action[0]] == 1 else "gold"
            i, j = action
            new_board[j] = new_board[i]
            new_board[i] = 0
            return "P{} moves {} from cell {} to {}: {}\n".format(player, coin, action[0], action[1], str(new_board))
