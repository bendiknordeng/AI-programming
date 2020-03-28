import numpy as np
from mcts import MonteCarloTreeSearch
from tree import Node
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class HexGame:
    neighbors = defaultdict(list)
    edges = defaultdict(list)
    all_moves = []

    def __init__(self, size, state=None, player=1):
        self.size = size
        self.state = state if state else self.generate_initial_state()
        if len(self.neighbors) == 0:
            self.__generate_neighbors()
        self.player = player

    def get_state(self):
        state = tuple(self.state.values())
        return (self.player, state)

    def reset(self):
        self.player = 1 # assuming player 1 always starts
        for cell in self.state:
            self.state[cell] = 0

    def sim_copy(self):
        return HexGame(self.size, self.state.copy(), self.player)

    @property
    def flat_state(self):
        flat_state = [self.player] + list(self.state.values())
        return np.asarray(flat_state, dtype = np.float64)

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
                self.all_moves.append((r,c))
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

    def result(self):
        if self.is_game_over()[0]:
            return 1 if 3-self.player == 1 else -1

    def is_game_over(self):
        for cell in self.edges[3-self.player][0]:
            if self.state[cell] == 3-self.player:
                winning_path = self.depth_first(cell, [cell])
                if winning_path:
                    return winning_path
        return False

    def depth_first(self, cell, path):
        for n in self.neighbors[cell]:
            if self.state[n] == 3-self.player and n not in path:
                path.append(n)
                if n in self.edges[3-self.player][1]:
                    return path
                else:
                    if self.depth_first(n, path):
                        return path
        return False

    def get_minimal_path(self, path):
        i = len(path)-1
        while True:
            if path[i] in self.edges[3-self.player][0]:
                if i != 0:
                    path = path[-1:i-1:-1]
                break
            temp_state = self.sim_copy()
            temp_state.state[path[i]] = 0
            if temp_state.is_game_over():
                self.state[path[i]] = 0
                path.remove(path[i])
            i-=1
        return path

    def move(self, action):
        """
        Input: action to be executed
        Returns: new state - use np.copy(self.state)
        """
        assert self.state[(action[0],action[1])] == 0, "Invalid move, cell not empty"
        self.state[(action[0],action[1])] = self.player
        self.player = 3 - self.player

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

    def draw(self, path=False, animation_delay = 0):
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
        if path:
            nx.draw(graph, pos=positions, nodelist=self.get_minimal_path(path), node_color='blue', node_size=1300-200*(self.size), ax=fig.axes[0])
            fig.axes[0].set_title("Player {} won".format(3-self.player))

        if animation_delay: # run animation automatically if delay > 0
            plt.show(block = False)
            plt.pause(animation_delay)
            plt.close()
        else: # show single figure if delay not given
            plt.show(block = True)

    def print_move(self, action):
        """
        Returns: string for verbose mode
        """
        return "Player {} put a piece on {}".format(self.player, action)

if __name__ == "__main__":
    game = HexGame(4)
    reds = [(0, 2), (1, 1), (1, 2), (2, 1),(3,1)]
    blacks = []
    for cell in reds:
        game.state[cell] = 1
    for cell in blacks:
        game.state[cell] = 2

    game.player = 2
    path = game.is_game_over()
    game.draw(path)
