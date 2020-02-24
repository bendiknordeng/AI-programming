import random
from tree import Node, Edge

class Game:
    def __init__(self, P, initialState):
        self.P = P  # starting player option
        self.turn = self.setStartingPlayer()
        self.root = Node(self.turn, initialState)

    def setStartingPlayer(self):
        if self.P == 1:
            return True
        elif self.P == 2:
            return False
        else:
            return random.random() >= 0.5

    def generateChildStates(self, node):
        for action in self.generateValidActions(node.state):
            child = Node(not node.turn, self.nextState(node.state, action), node)
            edge = Edge(action, node, child)
            child.setPrevAction(edge)
            node.addChild(edge, child)

    def getReinforcement(self, node):
        if not self.finalState(node.state): return 0
        if node.turn:
            return 1
        else:
            return -1

class NIM(Game):
    def __init__(self, P, N, K):
        super().__init__(P, N)
        self.K = K

    def nextState(self, pileCount, action):
        assert action <= self.K and pileCount - action >= 0, str(
            action) + " is not a valid amount of pieces. Max allowed is " + str(min(pileCount, self.K))
        return pileCount-action

    def generateValidActions(self, pileCount):
        return list(range(1, min(pileCount, self.K)+1))

    def finalState(self, pileCount):
        return pileCount == 0


class Ledge(Game):
    def __init__(self, P, board):
        super().__init__(P, board)
        assert board.count(
            2) == 1, "There can only be one gold coin on the board, you put " + str(board.count(2))
        self.boardLength = len(board)

    def nextState(self, board, action):
        if action == 0:
            assert board[0] != 0, 'There is no coin on the ledge'
            board[0] = 0
        else:
            i, j = board
            board[i] = board[j]
            board[j] = 0
        return board

    def generateValidActions(self, board):
        valid = []
        for i in range(self.boardLength-1):
            if board[i] != 0 and board[i + 1] == 0:  # non-empty cell with empty neighbor
                to = []
                j = i + 1
                while board[j] == 0:  # while there are empty cells to the right
                    to.append(j)
                    j += 1
                [valid.append((i,j)) for i in to]
        if board[0] != 0: valid.append(0)
        return valid

    def finalState(self, board):
        return board.count(2) == 0
