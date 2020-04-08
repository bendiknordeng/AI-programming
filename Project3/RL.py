import math
import random
import copy

import numpy as np
from ANN import ANN
from CNN import CNN
from game import HexGame
from matplotlib import pyplot as plt
from mcts import MonteCarloTreeSearch
from tqdm import tqdm

np.set_printoptions(linewidth=500)  # print formatting

class RL:
    def __init__(self, G, M, env, ANN, MCTS, save_interval, buffer_size, batch_size):
        self.G = G
        self.M = M
        self.env = env
        self.ANN = ANN
        self.MCTS = MCTS
        self.save_interval = save_interval
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.losses = []
        self.accuracies = []
        self.buffer = []
        self.all_cases = []

    def run(self):
        for i in tqdm(range(G)):
            if i % self.save_interval == 0:
                self.ANN.save(size=env.size, level=i)
                ANN.epochs += 10
                if i > 1:
                    self.plot(episode=i, save=True)
            self.env.reset()
            self.MCTS.init_tree()
            while not self.env.is_game_over():
                D = self.MCTS.search(self.env, self.M)
                self.add_case(D)
                env.move(env.all_moves[np.argmax(D)])
            batch_size = math.floor(len(self.all_cases)/2)
            training_cases = random.sample(self.all_cases, batch_size)
            x_train, y_train = list(zip(*training_cases))
            x_test, y_test = list(zip(*self.all_cases))
            self.train_ann(x_train, y_train, x_test, y_test)
            self.MCTS.eps *= 0.995
        self.ANN.save(size=env.size, level=i+1)
        self.write_db("cases/size_{}".format(self.env.size), self.buffer)
        self.plot(episode=G)

    def add_case(self, D):
        self.all_cases.append((env.flat_state, D))
        self.buffer.append((env.flat_state, D))
        if len(self.buffer) > 500:
            self.buffer.pop(0)

    def get_win_rate(self, p1, p2): # get win-rate for p1 in games vs p2
        game = HexGame(self.env.size)
        wins = np.zeros(100)
        for i in range(100):
            p1_starts = bool(i%2)
            game.reset()
            move = p1.get_greedy(game) if p1_starts else p2.get_greedy(game)
            game.move(move)
            turn = not p1_starts
            while not game.is_game_over():
                move = p1.get_greedy(game) if turn else p2.get_greedy(game)
                game.move(move)
                turn = not turn
            if (p1_starts and game.result() == 1) or (not p1_starts and game.result() == -1):
                wins[i] = 1
        return sum(wins)/100

    def train_ann(self, x_train, y_train, x_test, y_test):
        self.ANN.fit(x_train, y_train)
        loss, acc = self.ANN.get_status(x_test, y_test)
        self.losses.append(loss)
        self.accuracies.append(acc)

    def plot(self, episode, save=False):
        self.episodes = np.arange(episode)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.episodes, self.losses, color='tab:orange', label="Loss")
        ax.plot(self.episodes, self.accuracies, color='tab:blue', label="Accuracy")
        plt.legend()
        if save:
            plt.savefig("plots/size-{}-episode-{}".format(self.env.size, episode))
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()
        else:
            plt.show()

    def generate_cases(self):
        cases = []
        print("Generating training cases")
        for i in tqdm(range(self.G)):
            self.env.reset()
            self.MCTS.init_tree()
            while not self.env.is_game_over():
                D = self.MCTS.search(self.env, self.M)
                cases.append((self.env.flat_state, D))
                self.env.move(self.env.all_moves[np.argmax(D)])
            #MCTS.eps *= 0.99
        self.write_db("cases/test_size_{}".format(self.env.size), cases)

    def plot_level_accuracies(self, levels):
        cases = self.load_db("cases/size_{}".format(self.env.size))
        losses = []
        accuracies = []
        for l in levels:
            self.ANN.load(self.env.size, l)
            input, target = list(zip(*cases))
            losses.append(self.ANN.get_loss(input, target))
            accuracies.append(self.ANN.accuracy(input, target))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlabel("episodes")
        fig.axes[0].set_title("Size {}".format(self.env.size))
        ax.plot(levels, accuracies, color='tab:blue', label="Accuracy")
        ax.plot(levels, losses, color='tab:orange', label="Loss")
        plt.legend()
        plt.show()

    def play_game(self):
        self.env.reset()
        while True:
            _, _, index = self.ANN.get_move(self.env)
            self.env.move(self.env.all_moves[index])
            self.env.draw(animation_delay = 0.2)
            winning_path = self.env.is_game_over()
            if winning_path:
                break
        self.env.draw(path=winning_path)

    def write_db(self, filename, cases):
        inputs, targets = list(zip(*cases))
        np.savetxt(filename+'_inputs.txt', inputs)
        np.savetxt(filename+'_targets.txt', targets)
        print("Cases have been written to \n{}\n{}".format(filename+'_inputs.txt', filename+'_targets.txt'))

    def load_db(self, filename):
        inputs = np.loadtxt(filename+'_inputs.txt')
        targets = np.loadtxt(filename+'_targets.txt')
        cases = list(zip(inputs, targets))
        return cases


if __name__ == '__main__':
    # MCTS/RL parameters
    board_size = 4
    G = 10
    M = 500
    save_interval = 5
    buffer_size = 2000
    batch_size = 1000

    # ANN parameters
    activation_functions = ["sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.005  # learning rate
    H_dims = [120, 84]
    activation = activation_functions[2]
    optimizer = optimizers[3]
    epochs = 0

    ANN = ANN(board_size**2, H_dims, alpha, optimizer, activation, epochs)
    CNN = CNN(board_size, alpha, epochs, activation, optimizer)
    #CNN.load(size=board_size, level=100)
    MCTS = MonteCarloTreeSearch(CNN, c=1., eps=1, stoch_policy=True)
    env = HexGame(board_size)
    RL = RL(G, M, env, ANN, MCTS, save_interval, buffer_size, batch_size)

    # Run RL algorithm and plot results
    RL.run()
    RL.play_game()

    # Generate training cases
    #RL.generate_cases()

    # Plot model accuracies and losses
    #levels = np.arange(0, 251, 50)
    #RL.plot_level_accuracies(levels)
