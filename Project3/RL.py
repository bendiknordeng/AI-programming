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

def RL(G, M, env, ANN, save_interval):
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
            action, D = MCTS.search(env, M)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            env.move(action)
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
    return ANN.make_dict(cases[0], cases[1])

def play_game(dict, env, ANN, delay=-1, verbose=False):
    env.reset()
    j = 0
    while True:
        input = tuple(env.flat_state)
        #if env.player == 1:
        #    ANN.load(env.size, 0)
        #else:
        #    ANN.load(env.size, 0)
        probs, action = ANN.get_move(env)
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
        while True:
            action, D = MCTS.search(env, M)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            env.move(action)
            winning_path = env.is_game_over()
            if winning_path:
                #env.draw(path=winning_path)
                break
            #M = math.floor(M*0.9)
        MCTS.eps *= 0.99
    write_db("cases_with_ann_policy/size_{}_inputs.txt".format(env.size), cases[0])
    write_db("cases_with_ann_policy/size_{}_targets.txt".format(env.size), cases[1])


def write_db(filename, object):
    np.savetxt(filename, object)


def load_db(filename):
    return np.loadtxt(filename)


if __name__ == '__main__':
    # Game parameters
    board_size = 6
    env = HexGame(board_size)

    # MCTS/RL parameters
    episodes = 200
    simulations = 500
    save_interval = 50

    #training_batch_size = 100
    ann_save_interval = 10
    eps_decay = 1

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.005  # learning rate
    H_dims = [math.floor(2*(2+2*board_size**2)/3)+board_size**2]*3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 1
    ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    #inputs = load_db("cases/size_{}_inputs.txt".format(board_size))
    #targets = load_db("cases/size_{}_targets.txt".format(board_size))
    #dict = train_ann(inputs, targets, ann)
    #print("Accuracy: {:3f}\nLoss: {:3f}".format(ann.accuracy([inputs,targets]),ann.get_loss([inputs,targets])))
    #generate_cases(episodes, simulations, HexGame(board_size), ann)

    dict = RL(episodes, simulations, env, ann, save_interval)

    def play(dict = dict, env=env, ANN=ann):
        play_game(dict, env, ann, 0.2, verbose = True)
    play()
