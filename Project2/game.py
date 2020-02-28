import numpy as np


class GameState:
    def __init__(self, state, next_to_move=1):
        self.state = state
        self.next_to_move = next_to_move

    def game_result(self):
        if self.is_game_over():
            if self.next_to_move == 2:
                return 1
            elif self.next_to_move == 1:
                return -1
        return None


class NIMState:
    def __init__(self, state, K, next_to_move=1):
        super().__init__(state, next_to_move)
        self.K = K

    def is_game_over(self):
        return self.state == 0

    def move(self, action):
        new_state = np.copy(self.state)
        return NIMState(new_state, self.K, 3 - self.next_to_move)

    def get_legal_actions(self):
        return list(range(1, min(self.state, self.K) + 1))


class LedgeState(GameState):
    def __init__(self, state, next_to_move=1):
        super().__init__(state, next_to_move)

    def is_game_over(self):
        return self.state.count(2) == 0

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

        return LedgeState(new_board, 3 - self.next_to_move)

    def get_legal_actions(self):
        valid = []
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

# def print_move(self, node):
#    player = 1 if node.parent.turn else 2
#    action = node.prev_action
#    remaining = "Remaining stones = {:<2}".format(node.state)
#    stones = "{:<1} stones".format(
#        action) if action > 1 else "{:<2} stone".format(action)
#    print("{:<2}: Player {} selects {:>8}: {:>21}".format(
#        node.count_parents(), player, stones, remaining))

# def print_move(self, node):
#    turn = node.parent.turn
#    player = 1 if turn else 2
#    action = node.prev_action
#    if action == 0:
#        coin = "copper" if node.parent.state[0] == 1 else "gold"
#        print("{:<2}: P{} picks up {}: {}".format(
#            node.count_parents(), player, coin, str(node.state)))
#    else:
#        coin = "copper" if node.parent.state[action[0]] == 1 else "gold"
#        print("{:<2}: P{} moves {} from cell {} to {}: {}".format(
#            node.count_parents(), player, coin, action[0], action[1], str(node.state)))
