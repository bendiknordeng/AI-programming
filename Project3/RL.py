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
    def __init__(self, G, M, env, ANN, MCTS, save_interval, batch_size, buffer_size, buffer=None, test_data=None):
        self.G = G
        self.M = M
        self.env = env
        self.ANN = ANN
        self.MCTS = MCTS
        self.save_interval = save_interval
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.test_data = test_data
        if test_data:
            self.x_test, self.y_test = test_data
        self.buffer = buffer if buffer else []
        self.start_plot = 0


    def run(self, plot_interval=1):
        eps_decay = 0.05 ** (1./self.G) if self.G > 100 else 1
        for i in tqdm(range(801, self.G)):
            if i % plot_interval == 0 and self.start_plot:
                self.plot(save=True)
            if i % self.save_interval == 0:
                self.ANN.save(size=env.size, level=i)
                #write_db('size_6_comp', self.buffer)
            self.env.reset()
            self.MCTS.init_tree()
            while not self.env.is_game_over():
                D = self.MCTS.search(self.env, self.M)
                self.add_case(D)
                env.move(env.all_moves[np.argmax(D)])
            self.train_ann(i)
            self.MCTS.eps *= eps_decay
        self.ANN.save(size=env.size, level=i+1)
        self.plot(save=True)

    def add_case(self, D):
        state = self.env.flat_state
        self.buffer.append((state, D))
        if random.random() > 0.5:
            self.buffer.append(self.rotated(state, D))

    def rotated(self, state, D):
        size = self.env.size
        player = state[0]
        return (np.asarray([player] + list(state[:0:-1])), D[::-1])

    def train_ann(self, i):
        self.batch_size = len(self.buffer)//2
        if self.batch_size > 500: # start training when enough cases to not overfit small sample
            if not self.start_plot: self.start_plot = i
            training_cases = random.sample(self.buffer, self.batch_size)
            x_train, y_train = list(zip(*training_cases))
            loss, acc = self.ANN.fit(x_train, y_train)
            self.train_losses.append(loss)
            self.train_accuracies.append(acc)
            if self.test_data:
                loss, acc = self.ANN.evaluate(self.x_test, self.y_test)
                self.test_losses.append(loss)
                self.test_accuracies.append(acc)

    def plot(self, save=False):
        x = np.arange(len(self.train_accuracies))+self.start_plot
        fig = plt.figure(figsize=(12,5))
        title = 'Size: {}   M: {}   lr: {}   Epochs: {}   '.format(self.env.size, self.M, self.ANN.alpha, self.ANN.epochs)
        title += 'Batch size: {}   Buffer size: {}'.format(self.batch_size, len(self.buffer))
        fig.suptitle(title, fontsize=10)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title("Accuracy")
        ax.plot(x, self.train_accuracies, color='tab:green', label="Train")
        if self.test_data: ax.plot(x, self.test_accuracies, color='tab:blue', label="Test")
        plt.grid()
        plt.legend()
        ax = fig.add_subplot(gs[0,1])
        ax.set_title("Loss")
        ax.plot(x, self.train_losses, color='tab:orange', label="Train")
        if self.test_data: ax.plot(x, self.test_losses, color='tab:purple', label="Test")
        plt.legend()
        plt.grid()
        if save:
            plt.savefig("plots/size-{}-cont".format(self.env.size))
            plt.close()
        else:
            if episode==self.G:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()

    def play_game(self, top_moves=5):
        self.env.reset()
        while True:
            probs, _, index = self.ANN.get_move(self.env)
            self.env.move(self.env.all_moves[index])
            self.env.draw(animation_delay = 0.2)
            print("Player-{} to play".format(3-self.env.player))
            val = {i: p for i, p in enumerate(probs)}
            sorted_moves = {k: v for k, v in sorted(val.items(), key=lambda item: item[1])}
            print("Top-{} moves:".format(top_moves))
            for i in range(1,top_moves+1):
                move = list(sorted_moves.keys())[-i]
                print("{:>2}: {:>5.2f}%".format(move, sorted_moves[move] * 100))
            print()
            winning_path = self.env.is_game_over()
            if winning_path:
                break
        self.env.draw(path=winning_path)

    def pre_train(self, x, y, epochs):
        n = len(x)
        for i in tqdm(range(epochs)):
            loss, acc = self.ANN.fit(x, y)
            self.train_losses.append(loss)
            self.train_accuracies.append(acc)
            self.plot(save=True)
        print("Loss: {}\nAccuracy: {}".format(loss, acc))

    def generate_cases(self):
        cases = []
        for i in tqdm(range(self.G)):
            self.env.reset()
            self.MCTS.init_tree()
            while not self.env.is_game_over():
                D = self.MCTS.search(self.env, self.M)
                cases.append((self.env.flat_state, D))
                cases.append(self.rotated(self.env.flat_state, D))
                self.env.move(self.env.all_moves[np.argmax(D)])
        write_db('cases/test_size_{}'.format(self.env.size), cases)

def write_db(filename, cases):
    if len(cases) == 0: return
    inputs, targets = list(zip(*cases))
    np.savetxt(filename+'_inputs.txt', inputs)
    np.savetxt(filename+'_targets.txt', targets)
    print("Buffer written to file")

def load_db(filename):
    inputs = np.loadtxt(filename+'_inputs.txt')
    targets = np.loadtxt(filename+'_targets.txt')
    print("Buffer loaded from file")
    return inputs.astype(int), targets


if __name__ == '__main__':
    # MCTS/RL parameters
    board_size = 6
    G = 1000
    M = 2000
    save_interval = 50
    batch_size = 500
    buffer_size = 1000

    # ANN parameters
    activation_functions = ["Linear", "Sigmoid", "Tanh", "ReLU"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001  # learning rate
    H_dims = [32, 32]
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 1

    #ANN = ANN(board_size**2, H_dims, alpha, optimizer, activation, epochs)
    CNN = CNN(board_size, H_dims, alpha, epochs, activation, optimizer)
    CNN.load(size=board_size, level=800)
    cases = load_db('cases/size_{}_comp'.format(board_size))
    cases = list(zip(*cases))
    eps = 1 * (0.05 ** (1./1000)) ** 800
    MCTS = MonteCarloTreeSearch(CNN, c=1.4, eps=eps, stoch_policy=True)
    env = HexGame(board_size)
    RL = RL(G, M, env, CNN, MCTS, save_interval, batch_size, buffer_size, buffer=cases, test_data=None)

    #x, y = cases[10000:]
    #RL.pre_train(x, y, 25)
    #CNN.save(size=6, level=1200)

    # Run RL algorithm and plot results
    RL.run(plot_interval=10)
    #RL.play_game()
    #RL.generate_cases()
