from game import HexState
from RL import RL_algorithm
from ANN import ANN
import random

class TOPP:
    def __init__(self, players, board_size):
        self.players = players
        self.board_size = board_size

    def run_tournament(self, games):
        results = []
        for game in games:
            for player1 in players:
                for player2 in players:
                    if player1 != player2:
                        self.play_game(player1, player2)

    def play_game(self, player1, player2):
        game = HexState(self.board_size)
        turn = random.choice([1, 2])
        while not game.is_game_over():
            import pdb; pdb.set_trace()
            game = player1.move(game) if turn == 1 else player2.move(game)
            turn = 3-turn



class Agent:
    def __init__(self, ANN):
        self.ANN = ANN

    def move(self, game):
        return game.move(self.ANN.get_move(game.flat_state))


if __name__ == '__main__':
    # TOPP parameters
    games_to_be_played = 10
    board_size = 4

    # MCTS/RL parameters
    episodes = 2
    simulations = 500
    training_batch_size = 10
    ann_save_interval = 1
    eps = 1
    eps_decay = 0.95

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001 # learning rate
    hidden_layer_sizes = [16,16,16]
    io_layer_size = board_size * board_size # input and output layer sizes (always equal)
    activation_func = activation_functions[0]
    optimizer = optimizers[0]
    epochs = 10

    ANN = ANN(alpha, epochs, io_layer_size, hidden_layer_sizes, activation_func, optimizer)
    RL_algorithm(episodes, simulations, training_batch_size, board_size, ann_save_interval, ANN, eps, eps_decay)

    #players = []
    #for i in range(int(episodes//ann_save_interval)):
    #    players.append(ANN.model.load_weights('models/model_'+str(i*ann_save_interval)+'_simulations'))
    #tournament = TOPP(players, board_size)
    #tournament.run_tournament(games_to_be_played)
