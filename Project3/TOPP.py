from game import HexState
from RL import RL_algorithm
from ANN import ANN
from collections import defaultdict
import random

class TOPP:
    def __init__(self, players, board_size):
        self.players = players
        self.board_size = board_size

    def run_tournament(games, players):
        results = defaultdict(int)
        for player1 in players:
            for player2 in players:
                if player1 != player2:
                    for _ in range(games):
                        result = play_game(player1, player2)
                        results[player1] += int(result)
                        results[player2] += int(not result)
            players.remove(player1)
        print(results)

    def play_game(player1, player2, display = False):
        game = HexState(4)
        turn = random.choice([True,False])
        print("{} starts with red".format(player1 if turn else player2))
        while not game.is_game_over():
            game = player1.move(game) if turn else player2.move(game)
            game.draw(1)
            turn = not turn
        return game, player2 if turn else player1

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
    games_to_be_played = 10
    board_size = 4

    # MCTS/RL parameters
    episodes = 20
    simulations = 500
    training_batch_size = 10
    ann_save_interval = 5
    eps = 1
    eps_decay = 0.95

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001 # learning rate
    hidden_layer_sizes = [10,10,10]
    io_layer_size = board_size * board_size # input and output layer sizes (always equal)
    activation_func = activation_functions[3]
    optimizer = optimizers[0]
    epochs = 10

    model = ANN(io_layer_size, hidden_layer_sizes, None, alpha, epochs, activation_func, optimizer)
    RL_algorithm(episodes, simulations, training_batch_size, board_size, ann_save_interval, model, eps, eps_decay)
    players = []
    for level in level_models:
        players.append(Agent(level_models[level], level))
    tournament = TOPP(players, board_size)
    tournament.run_tournament(games_to_be_played)
