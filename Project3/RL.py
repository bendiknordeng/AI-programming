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


def RL_algorithm(games, simulations, env, ANN, eps_decay, epochs):
    cases = [[], []]
    MCTS = MonteCarloTreeSearch(ANN)
    for i in tqdm(range(games)):
        env.reset()
        MCTS.init_tree()
        M = simulations
        while not env.is_game_over():
            action, D = MCTS.search(env, M)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            env.move(action)
            #M = math.ceil(M*0.8)
        fit_cases = list(zip(cases[0], cases[1]))
        fit_cases = random.sample(fit_cases, math.ceil(len(cases[0]) / 2))
        ANN.fit(list(zip(*fit_cases)))
        ANN.epochs += math.floor(np.exp(i * 3 / (games)))
        # if (i+1) % save_interval == 0:
        #    ANN.model.save_weights(model_path.format(level=i+1))
    train_ann(cases[0], cases[1], ANN)
    """
    ANN.epochs = 1
    accuracies = []
    epochs = []
    split = math.floor(len(cases[0])/10)
    val_data = [cases[0][0:split], cases[1][0:split]] #10% of data is validation data
    train_data = list(zip(cases[0][split:len(cases[0])],cases[1][split:len(cases[1])]))
    print("fitting")
    for epoch in tqdm(range(1000)):
        random.shuffle(train_data)
        ANN.fit(list(zip(*train_data)))
        acc = ANN.accuracy(val_data)
        accuracies.append(acc)
        epochs.append(epoch)
    print("terminated after epoch number", epoch)
    plt.plot(epochs, accuracies)
    plt.show()
    return ANN.make_dict(cases)
    """


def play_game(dict, env, ANN, delay=-1, verbose=True):
    env.reset()
    j = 0
    while not env.is_game_over():
        input = env.flat_state
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
            env.draw(delay)
        env.move(action)
        j += 1
    winning_player = 3 - env.flat_state[0]
    print("player", winning_player, "won after", j, "moves.")
    if delay > -1:
        env.draw()


def say():
    import os
    os.system('say "gamle ørn, jeg er ferdig  "')


def generate_cases(games, simulations, env):
    cases = [[], []]
    MCTS = MonteCarloTreeSearch()
    for i in tqdm(range(games)):
        env.reset()
        MCTS.init_tree()
        M = simulations
        while not env.is_game_over():
            action, D = MCTS.search(env, M)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            env.move(action)
    return cases


def train_ann(inputs, targets, ANN):
    # Shuffle cases before training
    train_data = list(zip(inputs,targets))
    random.shuffle(train_data)
    inputs, targets = zip(*train_data)

    split = math.floor(len(inputs) * 0.01) # 1 percent of data is test
    test_data = [inputs[0:split], targets[0:split]]
    train_data = list(zip(inputs[split:len(inputs)],targets[split:len(targets)]))
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


def write_db(filename, object):
    np.savetxt(filename, object)


def load_db(filename):
    return np.loadtxt(filename)


if __name__ == '__main__':
    # Game parameters
    board_size = 4
    env = HexGame(board_size)

    # MCTS/RL parameters
    episodes = 1000
    simulations = 500

    #training_batch_size = 100
    ann_save_interval = 10
    eps_decay = 1

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.01  # learning rate
    H_dims = [math.floor(2*(1+board_size**2)/3)+board_size**2] * 3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 200
    ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)

    #cases = generate_cases(episodes, simulations, HexGame(board_size))
    #inputs = cases[0]
    #targets = cases[1]
    # write_db( , inputs)
    # write_db( , targets)

    inputs = load_db("size_four_inputs.txt")
    targets = load_db("size_four_targets.txt")
    pred_dict = train_ann(inputs, targets, ann)
    print("Accuracy: {:3f}\nLoss: {:3f}".format(ann.accuracy([inputs,targets]),ann.get_loss([inputs,targets])))

    def play(dict=pred_dict, env=env, ANN=ann):
        play_game(dict, env, ann, 0.1, 0)
    play()

    """
    for i in range(1):
        print(i)
        pred_dict = RL_algorithm(episodes, simulations, env, ann, eps_decay, epochs)

        play()
    """

    #import pdb; pdb.set_trace()
