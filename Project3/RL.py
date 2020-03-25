from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
import random
from tqdm import tqdm
import math
import numpy as np
np.set_printoptions(linewidth=160) # print formatting

def RL_algorithm(games, simulations, env, ANN, eps_decay, epochs):
    cases = []
    MCTS = MonteCarloTreeSearch(ANN)
    for i in tqdm(range(games)):
        env.reset()
        MCTS.init_tree()
        M = simulations
        while not env.is_game_over():
            action, D = MCTS.search(env, M)
            cases.append((env.flat_state,D))
            env.move(action)

            #M = math.ceil(M*0.5)
        fit_cases = random.sample(cases, math.ceil(len(cases)/2))
        _ = ANN.fit(0, fit_cases)

        ANN.epochs += math.ceil(10/float(games)) #increment epochs

        #if (i+1) % save_interval == 0:
        #    ANN.model.save_weights(model_path.format(level=i+1))

    #run through of training data
    accuracies = []
    ANN.epochs = 30
    interval = math.floor(len(cases)/4)
    runs = 0
    accuracy = 0
    while accuracy < 0.6 and runs < 30: #run 10 * 30 epochs
        runs += 1
        random.shuffle(cases) #shuffle data before each split
        for i in range(3): # leave out 25% of data
            start = i*interval
            end = ((i+1)*interval)
            fit_cases = cases[start:end]
            _ = ANN.fit(0, fit_cases)
        accuracy = ANN.accuracy(cases[3*interval:4*interval])
        accuracies.append(accuracy)
    print("terminated after", runs, "runs.")
    print("accuracy", accuracies)

    ANN.epochs = 1 # only to make dict of known cases
    dict = ANN.fit(games-1, cases)
    return dict


def play_game(dict, env, ANN, delay = -1,verbose=True):
    env.reset()
    inputs = []
    moves = []
    preds = []
    j = 0
    while not env.is_game_over():
        inputs.append(env.flat_state)
        probs, action = ANN.get_move(env)
        if verbose:
            print()
            input = tuple(env.flat_state)
            print(input)
            if dict.get(input) != None:
                for tar, pred in dict[input]:
                    print(tar, pred)
            else:
                print("No such case for input state")
            print()
            print(np.around(probs.numpy()*100, decimals = 1))
            if delay > -1:
                env.draw()
        else:
            if delay > -1:
                env.draw(delay)

        preds.append(np.around(probs.numpy()*100, decimals = 1))
        moves.append(action)
        env.move(action)
        j += 1
    winning_player =  3 - env.flat_state[0]
    print("player", winning_player, "won after", j, "moves.")

    """
    print()
    print()
    print("played game")
    for i in range(len(moves)):
        print()
        print(inputs[i])
        print(preds[i], "  ",moves[i])
    """
    if delay > -1:
        env.draw()


def say():
    import os
    os.system('say "gamle ørn, jeg er ferdig  "')


if __name__ == '__main__':
    # Game parameters
    board_size = 3

    # MCTS/RL parameters

    episodes = 30

    simulations = 1000

    #training_batch_size = 100
    ann_save_interval = 10
    eps_decay = 0.95

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001 # learning rate
    H_dims = [board_size, board_size**2]
    io_dim = board_size * board_size # input and output layer sizes (always equal)
    activation = activation_functions[3]
    optimizer = optimizers[3]

    epochs = 1

    for i in range(30):
        print(i)
        env = HexGame(board_size)
        ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
        prediction_dictionary = RL_algorithm(episodes, simulations, env, ann, eps_decay, epochs)
        def play(dict = prediction_dictionary, env = env, ANN = ann):
            play_game(dict, env,ann,-1,0)
        play()
    say()

    import pdb; pdb.set_trace()
