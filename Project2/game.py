import random
from tree import Node


class Game:
    def __init__(self, P, initial_state):
        self.P = P  # starting player option
        turn = self.set_starting_player()
        self.root = Node(turn, initial_state, is_root=True)

    def set_starting_player(self):
        if self.P == 1:
            return True
        elif self.P == 2:
            return False
        else:
            return random.random() >= 0.5

    def generate_children(self, node):
        num_child = 1
        for action in self.generate_valid_actions(node.state):
            child = Node(not node.turn, self.next_state(
                node.state, action), node, action, num_child)
            node.add_child(child)
            num_child += 1

    def get_reinforcement(self, node, starting_player):
        if node.parent.turn == starting_player.turn:
            return 1
        else:
            return 0


class NIM(Game):
    def __init__(self, P, N, K):
        super().__init__(P, N)
        self.K = K

    def next_state(self, pile_count, action):
        assert action <= self.K and pile_count - action >= 0, str(
            action) + " is not a valid amount of pieces. Max allowed is " + str(min(pile_count, self.K))
        return pile_count - action

    def generate_valid_actions(self, pile_count):
        return list(range(1, min(pile_count, self.K) + 1))

    def final_state(self, node):
        return node.state == 0

    def print_move(self, node):
        player = 1 if node.parent.turn else 2
        action = node.prev_action
        remaining = "Remaining stones = {:<2}".format(node.state)
        stones = "{:<1} stones".format(
            action) if action > 1 else "{:<2} stone".format(action)
        print("{:<2}: Player {} selects {:>8}: {:>21}".format(
            node.count_parents(), player, stones, remaining))


class Ledge(Game):
    def __init__(self, P, board):
        super().__init__(P, board)
        assert board.count(
            2) == 1, "There can only be one gold coin on the board, you put " + str(board.count(2))
        self.board_length = len(board)

    def next_state(self, board, action):
        temp_board = board.copy()
        if action == 0:
            assert temp_board[0] != 0, 'There is no coin on the ledge'
            temp_board[0] = 0
        else:
            i, j = action
            assert temp_board[i] != 0, 'There is no coin in spot {}'.format(i)
            assert temp_board[j] == 0, 'You cannot put a coin in spot {}'.format(
                j)
            temp_board[j] = temp_board[i]
            temp_board[i] = 0
        return temp_board

    def generate_valid_actions(self, board):
        valid = []
        for i in range(self.board_length - 1):
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

    def final_state(self, node):
        return node.state.count(2) == 0

    def print_move(self, node):
        turn = node.parent.turn
        player = 1 if turn else 2
        action = node.prev_action
        if action == 0:
            coin = "copper" if node.parent.state[0] == 1 else "gold"
            print("{:<2}: P{} picks up {}: {}".format(
                node.count_parents(), player, coin, str(node.state)))
        else:
            coin = "copper" if node.parent.state[action[0]] == 1 else "gold"
            print("{:<2}: P{} moves {} from cell {} to {}: {}".format(
                node.count_parents(), player, coin, action[0], action[1], str(node.state)))
