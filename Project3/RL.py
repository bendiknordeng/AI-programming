from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
import random
from tqdm import tqdm
import math
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=160) # print formatting

def RL_algorithm(games, simulations, env, ANN, eps_decay, epochs):
    cases = []
    MCTS = MonteCarloTreeSearch(ANN)
    epoch_increment = math.ceil(100/float(games))
    for i in tqdm(range(games)):
        env.reset()
        MCTS.init_tree()
        M = simulations
        while not env.is_game_over():
            action, D = MCTS.search(env, M)
            cases.append((env.flat_state,D))
            env.move(action)
            #M = math.ceil(M*0.8)
        fit_cases = random.sample(cases, math.ceil(len(cases)/2))
        ANN.fit(fit_cases)
        ANN.epochs += math.floor(np.exp(i/(games/3))) #increment epochs
        #if (i+1) % save_interval == 0:
        #    ANN.model.save_weights(model_path.format(level=i+1))
    #run through cases
    accuracies = []
    epochs = []
    ANN.epochs = 20
    split = math.floor(len(cases)/10)
    val_data = cases[0:split] #20% of data is validation data
    train_data = cases[split:len(cases)]
    train_split = math.floor(len(train_data)/4)
    runs = 0
    accuracy = 0
    accuracies.append(ANN.accuracy(val_data))
    epochs.append(ANN.epochs*runs)
    while accuracy < 0.5 and runs < 100: #run 10 * 30 epochs
        runs += 1
        random.shuffle(train_data) #shuffle data before each split
        for i in range(4): # train ann on train_data
            start = i*train_split
            end = ((i+1)*train_split)
            ANN.fit(train_data[start:end])
        accuracy = ANN.accuracy(val_data)
        accuracies.append(accuracy)
        epochs.append(ANN.epochs*runs)
    print("terminated after", runs, "runs.")
    plt.plot(epochs, accuracies)
    plt.show()
    return ANN.make_dict(cases)


def play_game(dict, env, ANN, delay=-1,verbose=True):
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
                targets = []
                for tar, _ in dict[input]:
                    targets.append(tar)
                mean_target = np.around(np.mean(targets, axis = 0),decimals = 1)
                print(mean_target)
            else:
                print("No such case for input state")
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
    if delay > -1:
        env.draw()


def say():
    import os
    os.system('say "gamle ørn, jeg er ferdig  "')


if __name__ == '__main__':
    # Game parameters
    board_size = 4

    # MCTS/RL parameters
    episodes = 100
    simulations = 1000

    #training_batch_size = 100
    ann_save_interval = 10
    eps_decay = 1

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.005 # learning rate
    H_dims = [board_size*board_size]
    io_dim = board_size * board_size # input and output layer sizes (always equal)
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 1

    for i in range(1):
        print(i)
        env = HexGame(board_size)
        ann = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
        pred_dict = RL_algorithm(episodes, simulations, env, ann, eps_decay, epochs)
        def play(dict = pred_dict, env = env, ANN = ann):
            play_game(dict, env,ann,-1,0)
        play()
    #say()

    import pdb; pdb.set_trace()
