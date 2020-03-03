import numpy as np

class HexBoardMove:
    def __init__(self, i, j, turn):
        self.i = i
        self.j = j
        self.turn = turn

    def __repr__(self):
        return "({},{})".format(self.i,self.j)

class HexState:
    neighbors = {}

    def __init__(self, size, state=None, turn=1):
        self.size = size
        self.state = state if state else self.generate_initial_state()
        if len(self.neighbors) == 0:
            self.generate_neighbors()
        self.turn = turn
        self.edges = []
        if self.turn == 2:
            self.edges.append([(i,0) for i in range(self.size)])
            self.edges.append([(i,self.size-1) for i in range(self.size)])
        else:
            self.edges.append([(self.size-1,i) for i in range(self.size)])
            self.edges.append([(0,i) for i in range(self.size)])

    def generate_initial_state(self):
        state = {}
        for i in range(self.size):
            for j in range(self.size):
                state[(i,j)] = 0 # empty cell
        return state

    def generate_neighbors(self):
        corner = self.size-1
        for i in range(self.size):
            for j in range(self.size):
                if i > 0 and j > 0 and i < corner and j < corner:
                    self.neighbors[(i,j)] = [(i,j-1),(i-1,j),(i-1,j+1),(i,j+1),(i+1,j),(i+1,j-1)]
                elif i == 0:
                    if j == 0:
                        self.neighbors[(i,j)] = [(i+1,j),(i,j+1)]
                    elif j == corner:
                        self.neighbors[(i,j)] = [(i,j-1),(i+1,j),(i+1,j-1)]
                    else:
                        self.neighbors[(i,j)] = [(i,j-1),(i,j+1),(i+1,j),(i+1,j-1)]
                else: # i == edge
                    if j == 0:
                        self.neighbors[(i,j)] = [(i-1,j),(i-1,j+1),(i,j+1),(i+1,j)]
                    elif j == corner:
                        self.neighbors[(i,j)] = [(i-1,j),(i,j-1)]
                    else:
                        self.neighbors[(i,j)] = [(i,j-1),(i-1,j),(i-1,j+1),(i,j+1)]

    @property
    def game_result(self):
        if self.is_game_over():
            if self.turn == 2:
                return 1
            elif self.turn == 1:
                return -1
        return None

    def is_game_over(self):
        """
        Returns: True if gamestate is game over
        """
        game_over = False
        for end in self.edges[1]:
            game_over = self.propegate_path(end)
            if game_over: break
        return game_over

    def propegate_path(self, cell, path=[]):
        for n in self.neighbors[cell]:
            if self.state[n] == 3-self.turn and n not in path:
                if n in self.edges[0]:
                    return True
                else:
                    path.append(n)
                    found = self.propegate_path(n, path)
                    if found: return True
        return False


    def move(self, action):
        """
        Input: action to be executed
        Returns: new state - use np.copy(self.state)
        """
        board = np.copy(self.state)
        board[(action.i,action.j)] = action.turn
        return HexState(self.size, board, 3-self.turn)

    def get_legal_actions(self):
        """
        Returns: list of valid actions for the board
        """
        valid_actions = []
        for cell in self.board:
            if board[cell] == 0:
                valid_actions.append(cell)
        return valid_actions

    @staticmethod
    def print_move(node, turn):
        """
        Returns: string for verbose mode
        """
        return "Player {} put a piece on {}".format(1 if turn else 2, node.prev_action)


if __name__ == "__main__":
    hex = HexState(4)
    #print(hex.state)
    new_state = hex.state.copy()
    cells = [(2,3),(2,1),(2,2),(3,1),(1,1),(1,0)]
    for cell in cells:
        new_state[cell] = 1
    hex = HexState(4, new_state, 2)
    print(hex.is_game_over())
    #print(HexState.neighbors[(0,1)])
