from tree import Tree
class MonteCarloTreeSearch:
    def __init__(self, c = 1):
        self.tree = Tree()
        self.c = c #exploration constant

    def initTree(self, board, game_mode):
        self.tree.setGameMode(game_mode)
        self.tree.stateToNode.clear()
        self.tree.addState(board.getState(), board.get_legal_actions())

    def search(self, board, simulations_number):
        self.board = board
        self.search_start_state = board.getState()
        for i in range(simulations_number):
            self.simulate(self.board, self.search_start_state)
        self.board.setPosition(self.search_start_state)
        return self.tree.treePolicy(self.search_start_state, 0) # find greedy best action

    def simulate(self, board, state):
        self.board.setPosition(state)
        traversedNodes = self.simTree(board)
        z = self.simDefault(board) #rollout board
        self.tree.backup(traversedNodes, z)

    def simTree(self, board):
        c = self.c
        path = [] #list of nodes traversed
        while not board.is_game_over():
            state = board.getState()
            if not self.tree.hasState(state):
                self.tree.addState(state, board.get_legal_actions())
                return path
            path.append(self.tree.getNode(state))
            action = self.tree.treePolicy(state, self.c) # find next action
            self.tree.getNode(state).setLastAction(action)
            board.move(action)
        return path

    def simDefault(self, board):
        while not board.is_game_over():
            action = self.tree.defaultPolicy(board)
            board.move(action)
        return board.player1Won()
