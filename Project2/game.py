import random
from tree import Node, Edge

class Game:
    def __init__(self, P, initial_state):
        self.P = P  # starting player option
        self.turn = self.set_starting_player()
        self.root = Node(self.turn, initial_state)

    def set_starting_player(self):
        if self.P == 1:
            return True
        elif self.P == 2:
            return False
        else:
            return random.random() >= 0.5

    def generate_child_states(self, node):
        for action in self.generate_valid_actions(node.state):
            child = Node(not node.turn, self.next_state(node.state, action), node)
            edge = Edge(action, node, child)
            child.set_prev_action(edge)
            node.add_child(edge, child)

    def get_reinforcement(self, node):
        if not self.final_state(node.state): return 0
        if node.turn:
            return 1
        else:
            return -1

class NIM(Game):
    def __init__(self, P, N, K):
        super().__init__(P, N)
        self.K = K

    def next_state(self, pile_count, action):
        assert action <= self.K and pile_count - action >= 0, str(
            action) + " is not a valid amount of pieces. Max allowed is " + str(min(pile_count, self.K))
        return pile_count-action

    def generate_valid_actions(self, pile_count):
        return list(range(1, min(pile_count, self.K)+1))

    def final_state(self, pile_count):
        return pile_count == 0


class Ledge(Game):
    def __init__(self, P, board):
        super().__init__(P, board)
        assert board.count(
            2) == 1, "There can only be one gold coin on the board, you put " + str(board.count(2))
        self.board_length = len(board)

    def next_state(self, board, action):
        if action == 0:
            assert board[0] != 0, 'There is no coin on the ledge'
            board[0] = 0
        else:
            i, j = board
            board[i] = board[j]
            board[j] = 0
        return board

    def generate_valid_actions(self, board):
        valid = []
        for i in range(self.board_length-1):
            if board[i] != 0 and board[i + 1] == 0:  # non-empty cell with empty neighbor
                to = []
                j = i + 1
                while board[j] == 0:  # while there are empty cells to the right
                    to.append(j)
                    j += 1
                [valid.append((i,j)) for i in to]
        if board[0] != 0: valid.append(0)
        return valid

    def final_state(self, board):
        return board.count(2) == 0
