import random


class Game:
    def __init__(self, G, M, P):
        self.G = G  # number of games in batch
        self.P = P  # starting player option
        self.M = M  # number of rollouts per game move
        self.turn = self.setStartingPlayer()

    def setStartingPlayer(self):
        if self.P == 1:
            return True
        elif self.P == 2:
            return False
        else:
            return random.random() >= 0.5


class NIM(Game):
    def __init__(self, G, M, P, N, K):
        super().__init__(G, P, M)
        self.N = N
        self.K = K

    def move(self, pieces):
        assert pieces <= self.K and self.N - pieces >= 0, str(
            pieces) + " is not a valid amount of pieces. Max allowed is " + str(min(self.N, self.K))
        self.N -= pieces
        self.turn = not self.turn

    def generateValidActions(self):
        return list(range(1, min(self.N, self.K)+1))


class Ledge(Game):
    def __init__(self, G, M, P, board):
        super().__init__(G, P, M)
        assert board.count(
            2) == 1, "There can only be one gold coin on the board, you put " + str(board.count(2))
        self.board = board
        self.boardLength = len(board)

    def move(self, action):
        if action == 0:
            assert self.board[0] != 0, 'There is no coin on the ledge'
            self.board[0] = 0
            self.turn = not self.turn
        else:
            i, j = action
            self.board[i] = self.board[j]
            self.board[j] = 0
            self.turn = not self.turn

    def generateValidActions(self):
        valid = []
        for i in range(self.boardLength-1):
            if self.board[i] != 0 and self.board[i + 1] == 0:  # non-empty cell with empty neighbor
                to = []
                j = i + 1
                while self.board[j] == 0:  # while there are empty cells to the right
                    to.append(j)
                    j += 1
                [valid.append((i,j)) for i in to]
        if self.board[0] != 0: valid.append(0)
        return valid


if __name__ == '__main__':
    G = 10
    M = 10
    P = 1
    N = 20
    K = 5
    B = [1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1]

    ledge = Ledge(G, M, P, B)
    nim = NIM(G, M, P, N, K)

    moves = ledge.generateValidActions()
    print(moves)
    print(ledge.board)
    ledge.move(moves[0])
    print(ledge.board)
