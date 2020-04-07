import math
import random

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
        self.ANN.save(env.size, 0)
        for i in tqdm(range(G)):
            self.env.reset()
            self.MCTS.init_tree()
            while not self.env.is_game_over():
                D = self.MCTS.search(self.env, self.M)
                self.add_case(D)
                env.move(env.all_moves[np.argmax(D)])
            #self.train_ann()
            self.train_cnn()
            if (i + 1) % self.save_interval == 0:
                self.save_model(level=i+1)
                self.ANN.epochs += 10
            self.MCTS.eps *= 0.99
        self.write_db("cases/size_{}".format(self.env.size), self.buffer)
        self.plot()

    def add_case(self, D):
        self.all_cases.append((env.flat_state, D))
        self.buffer.append((env.flat_state, D))
        if len(self.buffer) > 500:
            self.buffer.pop(0)

    def train_ann(self):
        training_cases = random.sample(self.buffer, min(len(self.buffer),self.batch_size))
        x_train, y_train = list(zip(*training_cases))
        self.ANN.fit(x_train, y_train)
        x_test, y_test = list(zip(*self.all_cases))
        loss = self.ANN.get_loss(x_test, y_test)
        accuracy = self.ANN.accuracy(x_test, y_test)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def train_cnn(self):
        training_cases = random.sample(self.buffer, min(len(self.buffer),self.batch_size))
        x_train, y_train = list(zip(*training_cases))
        loss, acc = self.ANN.fit(x_train, y_train)
        self.losses.append(loss)
        self.accuracies.append(acc)

    def save_model(self, level):
        self.ANN.save(size=env.size, level=level)

    def plot(self):
        self.episodes = np.arange(self.G)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.episodes, self.losses, color='tab:orange', label="Loss")
        ax.plot(self.episodes, self.accuracies, color='tab:blue', label="Accuracy")
        plt.legend()
        plt.show()

    def generate_cases(self):
        cases = []
        self.MCTS.eps = 1
        print("Generating training cases")
        for i in tqdm(range(self.G)):
            self.env.reset()
            self.MCTS.init_tree()
            while not self.env.is_game_over():
                D = self.MCTS.search(self.env, self.M)
                cases.append((self.env.flat_state, D))
                self.env.move(self.env.all_moves[np.argmax(D)])
            MCTS.eps *= 0.99
        self.write_db("cases/size_{}".format(self.env.size), cases)

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
        print("Buffer have been written to \n{}\n{}".format(filename+'_inputs.txt', filename+'_targets.txt'))

    def load_db(self, filename):
        inputs = np.loadtxt(filename+'_inputs.txt')
        targets = np.loadtxt(filename+'_targets.txt')
        cases = list(zip(inputs, targets))
        return cases


if __name__ == '__main__':
    # MCTS/RL parameters
    board_size = 4
    G = 30
    M = 500
    save_interval = 50
    buffer_size = 2000
    batch_size = 1000

    # ANN parameters
    activation_functions = ["sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001  # learning rate
    H_dims = [128, 128, 64, 64]
    activation = activation_functions[2]
    optimizer = optimizers[3]
    epochs = 10

    #ANN = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    ANN = CNN(board_size, alpha, epochs, activation, optimizer)
    MCTS = MonteCarloTreeSearch(ANN, c=1., eps=1, stoch_policy=True)
    env = HexGame(board_size)
    RL = RL(G, M, env, ANN, MCTS, save_interval, buffer_size, batch_size)
    #cases = RL.load_db('cases/size_5')
    #x_train, y_train = list(zip(*cases))
    #loss, acc = ANN.fit(x_train, y_train)
    #print("Loss: {}\nAcc: {}".format(loss,acc))
    #ANN.save(5,200)

    # Run RL algorithm and plot results
    RL.run()
    RL.play_game()

    # Generate training cases
    #RL.generate_cases()

    # Plot model accuracies and losses
    #levels = np.arange(0, 251, 50)
    #RL.plot_level_accuracies(levels)
