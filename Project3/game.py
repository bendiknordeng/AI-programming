import numpy as np
from mcts import MonteCarloTreeSearch
from tree import Node
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class HexState:
    neighbors = defaultdict(list)
    edges = defaultdict(list)

    def __init__(self, size, state=None, player=1):
        self.size = size
        self.state = state if state else self.generate_initial_state()
        if len(self.neighbors) == 0:
            self.__generate_neighbors()
        self.player = player

    def generate_initial_state(self):
        state = {}
        for r in range(self.size):
            for c in range(self.size):
                state[(r,c)] = 0 # empty cell
        return state

    def __generate_neighbors(self):
        edge = self.size-1
        for r in range(self.size):
            for c in range(self.size):
                if r == 0:
                    if c == 0:
                        self.neighbors[(r,c)] = [(r+1,c),(r,c+1)]
                    elif c == edge:
                        self.neighbors[(r,c)] = [(r,c-1),(r+1,c),(r+1,c-1)]
                    else:
                        self.neighbors[(r,c)] = [(r,c-1),(r,c+1),(r+1,c),(r+1,c-1)]
                elif r == edge:
                    if c == 0:
                        self.neighbors[(r,c)] = [(r-1,c),(r-1,c+1),(r,c+1)]
                    elif c == edge:
                        self.neighbors[(r,c)] = [(r-1,c),(r,c-1)]
                    else:
                        self.neighbors[(r,c)] = [(r,c-1),(r-1,c),(r-1,c+1),(r,c+1)]
                else:
                    if c == 0:
                        self.neighbors[(r,c)] = [(r-1,c),(r-1,c+1),(r,c+1),(r+1,c)]
                    elif c == edge:
                        self.neighbors[(r,c)] = [(r-1,c),(r,c-1),(r+1,c-1),(r+1,c)]
                    else:
                        self.neighbors[(r,c)] = [(r,c-1),(r-1,c),(r-1,c+1),(r,c+1),(r+1,c),(r+1,c-1)]

        self.edges[1].append([(0,c) for c in range(self.size)])
        self.edges[1].append([(self.size-1,c) for c in range(self.size)])
        self.edges[2].append([(r,0) for r in range(self.size)])
        self.edges[2].append([(r,self.size-1) for r in range(self.size)])

    @property
    def flat_state(self):
        state = []
        for cell in self.state:
            state.append(self.state[cell])
        return np.array([state])

    @property
    def all_moves(self):
        moves = []
        for cell in self.state:
            moves.append(cell)
        return moves

    @property
    def game_result(self):
        if self.is_game_over():
            return 3-self.player
        return None

    def is_game_over(self):
        for cell in self.edges[3-self.player][0]:
            if self.state[cell] == 3-self.player:
                if self.depth_first(cell, []):
                    return True
        return False

    def depth_first(self, cell, path):
        for n in self.neighbors[cell]:
            if self.state[n] == 3-self.player and n not in path:
                if n in self.edges[3-self.player][1]:
                    return True
                else:
                    path.append(n)
                    if self.depth_first(n, path):
                        return True
        return False

    def move(self, action):
        """
        Input: action to be executed
        Returns: new state - use np.copy(self.state)
        """
        board = self.state.copy()
        board[(action[0],action[1])] = self.player
        return HexState(self.size, board, 3-self.player)

    def get_legal_actions(self):
        """
        Returns: list of valid actions for the board
        """
        valid_actions = []
        for cell in self.state:
            if self.state[cell] == 0:
                valid_actions.append(cell)
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
        empty, reds, blacks = self.cell_states()
        positions = self.__cell_positions()
        nx.draw(graph, pos=positions, nodelist=empty, node_color='white', edgecolors='black', node_size=1300-100*(self.size), ax=fig.axes[0])
        nx.draw(graph, pos=positions, nodelist=reds, node_color='red', edgecolors='black', node_size=1300-100*(self.size), ax=fig.axes[0])
        nx.draw(graph, pos=positions, nodelist=blacks, node_color='black', edgecolors='black', node_size=1300-100*(self.size), ax=fig.axes[0])

        if animation_delay: # run animation automatically if delay > 0
            plt.show(block = False)
            plt.pause(animation_delay)
            plt.close()
        else: # show single figure if delay not given
            plt.show(block = True)

    @staticmethod
    def print_move(player, action):
        """
        Returns: string for verbose mode
        """
        return "Player {} put a piece on {}".format(player, action)

if __name__ == "__main__":
    state = HexState(4)
    node = Node(state)
    mcts = MonteCarloTreeSearch(node)
    action = mcts.best_action(500)
    import pdb; pdb.set_trace()
