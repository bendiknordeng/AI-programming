import numpy as np
from mcts import MonteCarloTreeSearch
from tree import Node
import networkx as nx
import matplotlib.pyplot as plt

class HexBoardMove:
    def __init__(self, cell, turn):
        self.i, self.j = cell
        self.turn = turn

    def __repr__(self):
        return "({},{})".format(self.i,self.j)

class HexState:
    neighbors = {}
    blue_edges = []
    red_edges = []

    def __init__(self, size, state=None, turn=1):
        self.size = size
        self.state = state if state else self.generate_initial_state()
        if len(self.neighbors) == 0:
            self.generate_neighbors()
        self.turn = turn
        if len(self.blue_edges) == 0:
            self.blue_edges.append([(i,0) for i in range(self.size)])
            self.blue_edges.append([(i,self.size-1) for i in range(self.size)])
            self.red_edges.append([(0,i) for i in range(self.size)])
            self.red_edges.append([(self.size-1,i) for i in range(self.size)])
        self.game_over = False

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
                        self.neighbors[(i,j)] = [(i-1,j),(i-1,j+1),(i,j+1)]
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
        if self.game_over: return True
        for cell in self.blue_edges[1]:
            if self.state[cell] == 1:
                self.game_over = self.__propegate_path(cell, 1)
            if self.game_over:
                return True
        for cell in self.red_edges[1]:
            if self.state[cell] == 2:
                self.game_over = self.__propegate_path(cell, 2)
            if self.game_over:
                return True
        return False

    def __propegate_path(self, cell, player, path=[]):
        for n in self.neighbors[cell]:
            if player == 1:
                if self.state[n] == 1 and n not in path:
                    if n in self.blue_edges[0]:
                        return True
                    else:
                        path.append(n)
                        found = self.__propegate_path(n, player, path)
                        if found: return True
            elif player == 2:
                if self.state[n] == 2 and n not in path:
                    if n in self.red_edges[0]:
                        return True
                    else:
                        path.append(n)
                        found = self.__propegate_path(n, player, path)
                        if found: return True
        return False

    def move(self, action):
        """
        Input: action to be executed
        Returns: new state - use np.copy(self.state)
        """
        board = self.state.copy()
        board[(action.i,action.j)] = action.turn
        return HexState(self.size, board, 3-self.turn)

    def get_legal_actions(self):
        """
        Returns: list of valid actions for the board
        """
        valid_actions = []
        for cell in self.state:
            if self.state[cell] == 0:
                valid_actions.append(HexBoardMove(cell,self.turn))
        return valid_actions

    def cell_states(self):
        cell_states = [[],[],[]]
        for cell in self.state:
            if self.state[cell] == 0:
                cell_states[0].append(cell)
            elif self.state[cell] == 1:
                cell_states[1].append(cell)
            elif self.state[cell] == 2:
                cell_states[2].append(cell)
        return cell_states

    def __cell_positions(self):
        positions = {}
        for (r,c) in self.state:
            positions[(r,c)] = (-10*r + 10*c, -20*r - 20*c)
        return positions

    def __cell_edges(self):
        edges = []
        for (r,c) in self.state:
            for (i,j) in self.state:
                if i == r+1 and j == c-1:
                    edges.append(((r,c),(i,j)))
                elif i == r+1 and j == c:
                    edges.append(((r,c),(i,j)))
                elif i == r and j == c+1:
                    edges.append(((r,c),(i,j)))
        return edges

    def draw(self, animation_delay = 0):
        graph = nx.Graph()
        graph.add_nodes_from([cell for cell in self.state])
        graph.add_edges_from(self.__cell_edges())
        fig = plt.figure(figsize = (9,7))
        plt.axes()
        empty, blues, reds = self.cell_states()
        positions = self.__cell_positions()
        nx.draw(graph, pos=positions, nodelist=empty, node_color='black', node_size=800, ax=fig.axes[0])
        nx.draw(graph, pos=positions, nodelist=blues, node_color='blue', node_size=800, ax=fig.axes[0])
        nx.draw(graph, pos=positions, nodelist=reds, node_color='red', node_size=800, ax=fig.axes[0])

        if animation_delay: # run animation automatically if delay > 0
            plt.show(block = False)
            plt.pause(animation_delay)
            plt.close()
        else: # show single figure if delay not given
            plt.show(block = True)

    @staticmethod
    def print_move(node, turn):
        """
        Returns: string for verbose mode
        """
        return "Player {} put a piece on {}".format(1 if turn else 2, node.prev_action)

if __name__ == "__main__":
    hex = HexState(4)
    new_state = hex.state.copy()
    red_cells = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (3, 0)]
    for cell in red_cells:
        new_state[cell] = 2

    blue_cells = [(0, 0), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    for cell in blue_cells:
        new_state[cell] = 1

    hex = HexState(4, new_state, 2)
    hex.draw()
    import pdb; pdb.set_trace()
    print(hex.game_result)

    #state = HexState(4)
    #node = Node(state)
    #mcts = MonteCarloTreeSearch(node)
    #action = mcts.best_action(500)
    #import pdb; pdb.set_trace()
