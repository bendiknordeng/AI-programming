from game import HexGame
from RL import RL
from ANN import ANN
from collections import defaultdict
import random
import math

class TOPP:
    def __init__(self, players1, players2, board_size):
        self.players1 = players1
        self.players2 = players2
        self.board_size = board_size
        self.env = HexGame(board_size)

    def run_tournament(self):
        print(); print()
        print("Tournament on a board size", self.board_size )

        for p1 in self.players1:
            print()
            for p2 in self.players2:
                winner_stats = self.play_game(self.players1[p1], self.players2[p2])
                print("Player lvl", p1, "- player lvl", p2,": player", winner_stats )

        for p1 in self.players2:
            print()
            for p2 in self.players1:
                winner_stats = self.play_game(self.players2[p1], self.players1[p2])
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
    board_size = 6

    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.01  # learning rate
    H_dims = [math.floor(2*(1+board_size**2)/3)+board_size**2] * 3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 10


    models = [0, 1, 50, 51, 100, 101, 150, 151, 200, 201]
    players1 = {}
    players2 = {}


    for i in range(0,len(models),2):
        ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
        ann.load(board_size, models[i])
        players1[models[i]] = ann
        ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
        ann.load(board_size, models[i+1])
        players2[models[i+1]] = ann
    tournament = TOPP(players1, players2, board_size)
    tournament.run_tournament()
