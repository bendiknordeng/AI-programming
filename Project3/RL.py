from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
import random
from tqdm import tqdm
import math
import numpy as np
from matplotlib import pyplot as plt
import copy
import time
np.set_printoptions(linewidth=160)  # print formatting

def RL(G, M, env, ANN, save_interval, use_dist):
    ANN.save(size=env.size, level=0)
    losses = []
    accuracies = []
    episodes = np.arange(G)
    cases = [[],[]]
    MCTS = MonteCarloTreeSearch(ANN)
    for i in tqdm(range(G)):
        env.reset()
        MCTS.init_tree()
        while not env.is_game_over():
            D = MCTS.search(env, M)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            if use_dist:
                indices = np.arange(env.size**2)
                index = np.random.choice(indices, p=D)
                env.move(env.all_moves[index])
            else:
                index = np.argmax(D)
                env.move(env.all_moves[index])
        training_cases = list(zip(cases[0],cases[1]))
        random.shuffle(training_cases)
        inputs, targets = zip(*training_cases)
        split = math.floor(len(inputs)/2)
        losses.append(ANN.fit([inputs[:split],targets[:split]], debug = True))
        accuracies.append(ANN.accuracy([inputs[split:],targets[split:]]))
        if (i+1) % save_interval == 0:
            ANN.save(size=env.size, level=i+1)
            ANN.epochs += 5
        MCTS.eps *= 0.99
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(episodes, losses, color='tab:orange', label="Loss")
    ax.plot(episodes, accuracies, color='tab:blue', label="Accuracy")
    plt.legend()
    plt.show()
    write_db("cases/size_{}_inputs_ANN.txt".format(env.size), cases[0])
    write_db("cases/size_{}_targets_ANN.txt".format(env.size), cases[1])
    return ANN.make_dict(cases[0], cases[1])

def train_ann(inputs, targets, ANN):
    # Shuffle cases before training
    train_data = list(zip(inputs,targets))
    random.shuffle(train_data)
    inputs, targets = zip(*train_data)

    split = math.floor(len(inputs) * 0.01) # 1 percent of data is test
    test_data = [inputs[:split], targets[:split]]
    train_data = list(zip(inputs[split:],targets[split:]))
    k = 5
    split = math.floor(len(train_data)/k)
    accuracies = [ANN.accuracy(test_data)]
    losses = [ANN.get_loss(test_data)]
    epochs = [0]
    # try to do k-fold cross validation
    print("Fitting...")
    for i in range(1,k+1):#tqdm(range(k-1)):
        print("Fold:", i)
        fit_data = train_data[(i-1)*split:i*split]
        losses.append(ANN.fit(list(zip(*fit_data)), debug = True))
        acc = ANN.accuracy(test_data)
        accuracies.append(acc)
        epochs.append(ANN.epochs * i)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, accuracies, color='tab:blue', label="Accuracy")
    ax.plot(epochs, losses, color='tab:orange', label="Loss")
    plt.legend()
    plt.show()
    return ANN.make_dict(inputs, targets)

def play_game(dict, env, ANN, delay=-1, verbose=False):
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
        if delay > -1:
            env.draw(animation_delay=delay)
        env.move(action)
        j += 1
        winning_path = env.is_game_over()
        if winning_path:
            break
    winning_player = 3 - env.player
    print("player", winning_player, "won after", j, "moves.")
    if delay > -1:
        env.draw(path=winning_path)

def say():
    import os
    os.system('say "gamle ørn, jeg er ferdig"')

def generate_cases(games, simulations, env, ann):
    cases = [[], []]
    MCTS = MonteCarloTreeSearch(ANN=ann)
    for i in tqdm(range(games)):
        env.reset()
        MCTS.init_tree()
        M = simulations
        while not env.is_game_over():
            D = MCTS.search(env, M)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            indices = np.arange(env.size**2)
            i = np.random.choice(indices, p=D)
            env.move(env.all_moves[i])
        #MCTS.eps *= 0.99
    write_db("cases/size_{}_inputs.txt".format(env.size), cases[0])
    write_db("cases/size_{}_targets.txt".format(env.size), cases[1])

def plot_model_accuracies(ann, size, cases):
    losses = []
    accuracies = []
    episodes = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    for e in episodes:
        ann.load(size, e)
        losses.append(ann.get_loss(cases))
        accuracies.append(ann.accuracy(cases))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(episodes, accuracies, color='tab:blue', label="Accuracy")
    ax.plot(episodes, losses, color='tab:orange', label="Loss")
    plt.legend()
    plt.show()


def write_db(filename, object):
    np.savetxt(filename, object)


def load_db(filename):
    return np.loadtxt(filename)


if __name__ == '__main__':
    # Game parameters
    board_size = 6
    env = HexGame(board_size)

    # MCTS/RL parameters
    episodes = 400
    simulations = 1000
    save_interval = 50

    #training_batch_size = 100
    eps_decay = 1
    use_dist = True

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.005  # learning rate
    H_dims = [math.floor(2*(2+2*board_size**2)/3)+board_size**2]*3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 5
    ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    #inputs = load_db("cases/size_{}_inputs_ANN.txt".format(board_size))
    #targets = load_db("cases/size_{}_targets_ANN.txt".format(board_size))
    #dict = train_ann(inputs, targets, ann)
    #print("Accuracy: {:3f}\nLoss: {:3f}".format(ann.accuracy([inputs,targets]),ann.get_loss([inputs,targets])))
    #ann.fit([inputs,targets])
    #generate_cases(episodes, simulations, HexGame(board_size), ann)

    dict = RL(episodes, simulations, env, ann, save_interval, use_dist)
    #plot_model_accuracies(ann, board_size, [inputs, targets])

    #def play(dict = dict, env=env, ANN=ann):
    #    play_game(dict, env, ann, 0.2, verbose = True)
    #play()
