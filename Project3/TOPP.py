import math
import random

import numpy as np
from CNN import CNN
from game import HexGame
from tqdm import tqdm


class TOPP:
    def __init__(self, players1, players2, board_size, num_games, stoch_percent):
        self.players1 = players1
        self.players2 = players2
        self.board_size = board_size
        self.env = HexGame(board_size)
        self.table = {p: 0 for p in self.players1}
        self.num_games = num_games
        self.stoch_percent = stoch_percent

    def run_tournament(self, display, print_games=False):
        print("Tournament on board size", self.board_size)
        if print_games:
            runlist = range(self.num_games)
        else:
            runlist = tqdm(range(self.num_games))
        for _ in runlist:
            for p1 in self.players1:
                game_print = ""
                for p2 in self.players2:
                    if p1 != p2:
                        if print_games: game_print += "{} - {}: ".format(p1,p2)
                        p1_won, moves = self.play_game(self.players1[p1], self.players2[p2], display)
                        winner = p1 if p1_won else p2
                        self.table[winner] += 1
                        if print_games: game_print += "{} won after {} moves.\n".format(winner, moves)
                if print_games: print(game_print)
        print("\nFinal results:")
        sorted_table = {player: result for player, result in sorted(self.table.items(), key=lambda item: item[1], reverse=True)}
        place = 1
        for player in list(sorted_table.keys()):
            print("{:>2}: {:>3}  - {:>2} wins".format(place, player, self.table[player]))
            place += 1

    def play_game(self, ann1, ann2, display=True):
        self.env.reset()
        moves = 0
        while not self.env.is_game_over():
            moves += 1
            _, stoch_index, greedy_index = ann1.get_move(self.env) if self.env.player == 1 else ann2.get_move(self.env)
            self.env.move(self.env.all_moves[stoch_index if random.random() < self.stoch_percent else greedy_index])
            if display: self.env.draw(animation_delay=0.2)
        p1_won = True if self.env.result() == 1 else False
        if display:
            path = self.env.is_game_over()
            self.env.draw(path=path, animation_delay=0.2)
        return p1_won, moves


if __name__ == '__main__':
    board_size = 6

    activation_functions = ["Linear", "Sigmoid", "Tanh", "ReLU"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001  # learning rate
    H_dims = [32, 32]
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 10

    num_games = 50
    bottom_level = 0
    top_level = 800
    interval = 50
    stoch_percent = 1.

    l = np.arange(bottom_level, top_level+1, interval)
    models = np.sort(np.concatenate([l,l]))
    players1 = {}
    players2 = {}

    for i in range(0,len(models),2):
        ann = CNN(board_size, H_dims)
        ann.load(board_size, models[i])
        players1[models[i]] = ann
        ann = CNN(board_size, H_dims)
        ann.load(board_size, models[i+1])
        players2[models[i+1]] = ann
    tournament = TOPP(players1, players2, board_size, num_games, stoch_percent)
    tournament.run_tournament(display=False, print_games=False)
