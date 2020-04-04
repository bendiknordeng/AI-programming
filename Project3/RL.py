import math
import random

import numpy as np
from ANN import ANN
from game import HexGame
from matplotlib import pyplot as plt
from mcts import MonteCarloTreeSearch
from tqdm import tqdm

np.set_printoptions(linewidth=160)  # print formatting


def RL(G, M, env, ANN, MCTS, save_interval):
    ann.save(env.size, 0)
    losses = []
    accuracies = []
    episodes = np.arange(G)
    cases = [[],[]]
    for i in tqdm(range(G)):
        env.reset()
        MCTS.init_tree()
        while not env.is_game_over():
            D = MCTS.search(env, M)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            index = np.argmax(D)
            env.move(env.all_moves[index])
        training_cases = list(zip(cases[0], cases[1]))
        random.shuffle(training_cases)
        inputs, targets = zip(*training_cases)
        split = math.floor(len(inputs) / 2)
        losses.append(ANN.fit([inputs[:split], targets[:split]], debug=True))
        accuracies.append(ANN.accuracy([inputs[split:], targets[split:]]))
        if (i + 1) % save_interval == 0:
            ANN.save(size=env.size, level=i+1)
            write_db("cases/size_{}_inputs_ANN.txt".format(env.size), cases[0])
            write_db("cases/size_{}_targets_ANN.txt".format(env.size), cases[1])
            ANN.epochs += 5
        MCTS.eps *= 0.99
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(episodes, losses, color='tab:orange', label="Loss")
    ax.plot(episodes, accuracies, color='tab:blue', label="Accuracy")
    plt.legend()
    plt.show()
    return ANN.make_dict(cases[0], cases[1])


def play_game(dict, env, ANN, delay=0., verbose=False):
    env.reset()
    j = 0
    while True:
        input = tuple(env.flat_state)
        probs, action, _ = ANN.get_move(env)
        if verbose:
            print()
            print(input)
            if dict.get(input) != None:
                targets = []
                for tar, _ in dict[input]:
                    targets.append(tar)
                mean_target = np.around(np.mean(targets, axis=0), decimals=1)
                print(mean_target)
            else:
                print("No such case for input state")
            print(np.around(probs.numpy() * 100, decimals=1))
        if delay:
            env.draw(animation_delay=delay)
        env.move(action)
        j += 1
        winning_path = env.is_game_over()
        if winning_path:
            break
    winning_player = 3 - env.player
    print("player", winning_player, "won after", j, "moves.")
    if delay:
        env.draw(path=winning_path)


def generate_cases(games, simulations, env, ann, MCTS):
    cases = [[], []]
    print("Generating training cases")
    for i in tqdm(range(games)):
        env.reset()
        MCTS.init_tree()
        M = simulations
        while not env.is_game_over():
            D = MCTS.search(env, M)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            indices = np.arange(env.size ** 2)
            i = np.random.choice(indices, p=D)
            env.move(env.all_moves[i])
        # MCTS.eps *= 0.99
    write_db("cases/size_{}_inputs.txt".format(env.size), cases[0])
    write_db("cases/size_{}_targets.txt".format(env.size), cases[1])


def plot_model_accuracies(ann, size, cases, levels):
    losses = []
    accuracies = []
    for l in levels:
        ann.load(size, l)
        losses.append(ann.get_loss(cases))
        accuracies.append(ann.accuracy(cases))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel("episodes")
    fig.axes[0].set_title("Size {}".format(size))
    ax.plot(levels, accuracies, color='tab:blue', label="Accuracy")
    ax.plot(levels, losses, color='tab:orange', label="Loss")
    plt.legend()
    plt.show()


def write_db(filename, object):
    np.savetxt(filename, object)


def load_db(filename):
    return np.loadtxt(filename)


if __name__ == '__main__':
    # Game parameters
    board_size = 4
    env = HexGame(board_size)

    # MCTS/RL parameters
    episodes = 10
    simulations = 500
    save_interval = 50

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.005  # learning rate
    H_dims = [math.floor(2 * (2 + 2 * board_size ** 2) / 3) + board_size ** 2] * 3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 5
    ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    mcts = MonteCarloTreeSearch(ann, c=1., eps=1, stoch_policy=True)

    # Generate training cases
    #generate_cases(episodes, simulations, HexGame(board_size), ann, mcts)

    # Plot model accuracies and losses
    #inputs = load_db("cases/size_{}_inputs_ANN.txt".format(board_size))
    #targets = load_db("cases/size_{}_targets_ANN.txt".format(board_size))
    #levels = np.arange(0, 301, 50)
    #plot_model_accuracies(ann, board_size, [inputs, targets], levels)

    # Run RL algorithm and play game with final model
    dict = RL(episodes, simulations, env, ann, mcts, save_interval)
    play_game(dict, env, ann, delay=0.2, verbose=False)
