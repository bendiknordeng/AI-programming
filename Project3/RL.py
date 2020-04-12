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
    def __init__(self, G, M, env, ANN, MCTS, save_interval, batch_size, buffer_size):
        self.G = G
        self.M = M
        self.env = env
        self.ANN = ANN
        self.MCTS = MCTS
        if self.G >= 200:
            self.eps_decay = 0.05 ** (1./self.G) # 5% random at end of run
        else:
            self.eps_decay = 0.99
        self.save_interval = save_interval
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.losses = []
        self.accuracies = []
        self.buffer = []

    def run(self, live_plot=False, plot_interval=1):
        for i in tqdm(range(self.G)):
            if live_plot and i % plot_interval == 0:
                self.plot(episode=i, save=True)
            if i % self.save_interval == 0:
                self.ANN.save(size=env.size, level=i)
            self.env.reset()
            self.MCTS.init_tree()
            while not self.env.is_game_over():
                D = self.MCTS.search(self.env, self.M)
                self.add_case(D)
                env.move(env.all_moves[np.argmax(D)])
            self.train_ann()
            self.MCTS.eps *= self.eps_decay
        self.ANN.save(size=env.size, level=i+1)
        self.plot(episode=i+1, save=True)

    def add_case(self, D):
        state = self.env.flat_state
        size = self.env.size
        self.buffer.append((state, D))
        if len(self.buffer) > self.buffer_size: self.buffer.pop(0)
        if random.random() > 0.5:
            player = state[0]
            state = state[1:].reshape(size, size)
            rot_state = np.rot90(state,k=2,axes=(0,1))
            probs = D.reshape(size, size)
            rot_D = np.rot90(probs, k=2, axes=(0,1))
            self.buffer.append((np.asarray([player] + list(rot_state.reshape(size**2))), rot_D.reshape(size**2)))
            if len(self.buffer) > self.buffer_size: self.buffer.pop(0)

    def train_ann(self):
        batch_size = min(self.batch_size,len(self.buffer))
        training_cases = random.sample(self.buffer, batch_size)
        x_train, y_train = list(zip(*training_cases))
        loss, acc = self.ANN.fit(x_train, y_train)
        self.losses.append(loss)
        self.accuracies.append(acc)

    def plot(self, episode, save=False):
        self.episodes = np.arange(episode)
        fig = plt.figure(figsize=(12,5))
        title = 'Size: {}   M: {}   lr: {}   Epochs: {}   '.format(self.env.size, self.M, self.ANN.alpha, self.ANN.epochs)
        title += 'Batch size: {}   Buffer size: {}'.format(self.batch_size, len(self.buffer))
        fig.suptitle(title, fontsize=10)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title("Accuracy")
        ax.plot(self.episodes, self.accuracies, color='tab:green', label="Batch")
        plt.grid()
        plt.legend()
        ax = fig.add_subplot(gs[0,1])
        ax.set_title("Loss")
        ax.plot(self.episodes, self.losses, color='tab:red', label="Binary cross entropy")
        plt.legend()
        plt.grid()
        if save:
            plt.savefig("plots/size-{}".format(self.env.size))
            plt.close()
        else:
            if episode==self.G:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()

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

if __name__ == '__main__':
    # MCTS/RL parameters
    board_size = 6
    G = 200
    M = 1000
    save_interval = 50
    batch_size = 128
    buffer_size = 500

    # ANN parameters
    activation_functions = ["Sigmoid", "Tanh", "ReLU"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001  # learning rate
    H_dims = [120, 84]
    activation = activation_functions[0]
    optimizer = optimizers[3]
    epochs = 10

    ANN = ANN(board_size**2, H_dims, alpha, optimizer, activation, epochs)
    CNN = CNN(board_size, alpha, epochs, activation, optimizer)
    #CNN.load(size=board_size, level=50)
    MCTS = MonteCarloTreeSearch(CNN, c=1.4, eps=1, stoch_policy=True)
    env = HexGame(board_size)
    RL = RL(G, M, env, CNN, MCTS, save_interval, batch_size, buffer_size)

    # Run RL algorithm and plot results
    RL.run(live_plot=True, plot_interval=10)
    RL.play_game()
