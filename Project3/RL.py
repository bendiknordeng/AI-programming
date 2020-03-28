from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
import random
from tqdm import tqdm
import math
import numpy as np
from matplotlib import pyplot as plt
import copy
np.set_printoptions(linewidth=160)  # print formatting

def play_game(dict, env, ANN, delay=-1, verbose=True):
    env.reset()
    j = 0
    while True:
        input = tuple(env.flat_state)
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
    winning_player = 3 - env.flat_state[0]
    print("player", winning_player, "won after", j, "moves.")
    if delay > -1:
        env.draw(path=winning_path)


def say():
    import os
    os.system('say "gamle ørn, jeg er ferdig  "')


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
        #MCTS.eps *= 0.99
    write_db("cases/size_{}_inputs.txt".format(env.size), cases[0])
    write_db("cases/size_{}_targets.txt".format(env.size), cases[1])


def train_ann(inputs, targets, ANN):
    # Shuffle cases before training
    train_data = list(zip(inputs,targets))
    random.shuffle(train_data)
    input, target = zip(*train_data)
    print("Fitting...")
    epochs, losses, accuracies = ANN.fit(input, target, debug = True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, accuracies, color='tab:blue', label="Accuracy")
    ax.plot(epochs, losses, color='tab:orange', label="Loss")
    plt.legend()
    plt.show()
    return ANN.make_dict(inputs, targets)


def write_db(filename, object):
    np.savetxt(filename, object)


def load_db(filename):
    return np.loadtxt(filename)


if __name__ == '__main__':
    # Game parameters
    board_size = 6
    env = HexGame(board_size)

    # MCTS/RL parameters
    episodes = 100
    simulations = 500

    #training_batch_size = 100
    ann_save_interval = 10
    eps_decay = 1

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001  # learning rate
    H_dims = [math.floor(2*(1+board_size**2)/3)+board_size**2] * 3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 100
    ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    inputs = load_db("cases/size_{}_inputs.txt".format(board_size))
    targets = load_db("cases/size_{}_targets.txt".format(board_size))
    pred_dict = train_ann(inputs, targets, ann)
    print("Accuracy: {:3f}\nLoss: {:3f}".format(ann.accuracy([inputs,targets]),ann.get_loss([inputs,targets])))

    def play(dict=pred_dict, env=env, ANN=ann):
        play_game(dict, env, ann, 0.1, 0)
    play()

    #generate_cases(episodes, simulations, HexGame(board_size), ann)
