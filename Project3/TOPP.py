from game import HexGame
from RL import RL
from ANN import ANN
from collections import defaultdict
import random
import math

class TOPP:
    def __init__(self, players, board_size):
        self.players = players
        self.board_size = board_size
        self.env = HexGame(board_size)

    def run_tournament(self):
        print(); print()
        print("Tournament on a board size", self.board_size )
        for p1 in self.players:
            print()
            for p2 in self.players:
                if self.players[p1] != self.players[p2]:
                    winner_stats = self.play_game(self.players[p1], self.players[p2])
                    print("Player lvl", p1, "- player lvl", p2,": player", winner_stats )
    def play_game(self, ann1, ann2, display = False):
        self.env.reset()
        j = 0
        while not self.env.is_game_over():
            j += 1
            _, action = ann1.get_move(self.env) if self.env.player == 1 else ann2.get_move(self.env)
            self.env.move(action)
            if display: self.env.draw(0.2)
        p1_won = True if self.env.result() == 1 else False
        if p1_won:
            return "1 won, " + str(j) + " moves."
        else:
            return "2 won, " + str(j) + " moves."

class Agent:
    def __init__(self, ANN, training_level):
        self.ANN = ANN
        self.training_level = training_level

    def move(self, game):
        return game.move(self.ANN.get_move(game))

    def __repr__(self):
        return "ANN_level_{}".format(self.training_level)


if __name__ == '__main__':
    # TOPP parameters
    board_size = 5

    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.01  # learning rate
    H_dims = [math.floor(2*(1+board_size**2)/3)+board_size**2] * 3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[0]
    epochs = 10


    models = [0,50,100,150,200]
    players = {}
    for level in models:
        ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
        ann.load(board_size, level)
        players[level] = ann
    tournament = TOPP(players, board_size)
    tournament.run_tournament()
