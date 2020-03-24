from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
import random
from tqdm import tqdm
import math
import numpy as np


def RL_algorithm(games, simulations, env, ANN, eps_decay):
    cases = []
    MCTS = MonteCarloTreeSearch(ANN)
    for i in range(games):#tqdm(range(games)):
        env.reset()
        MCTS.init_tree()
        M = simulations
        while not env.is_game_over():
            action, D = MCTS.search(env, M)
            cases.append((env.flat_state,D))
            D = np.around(np.asarray(D), decimals = 3)
            print(D)
            env.move(action)
            env.draw()
            M = math.ceil(M*0.5)
        fit_cases = random.sample(cases, math.ceil(len(cases)/math.floor(math.sqrt(i+1))))
        ANN.fit(random.sample(cases, math.ceil(len(cases)/math.floor(math.sqrt(i+1)))))
        #if (i+1) % save_interval == 0:
        #    ANN.model.save_weights(model_path.format(level=i+1))
        MCTS.eps *= eps_decay
    #say()

def play_game(env, ANN, delay=0,verbose=True):
    env.reset()
    while not env.is_game_over():
        if verbose: print(ANN.forward(env.flat_state))
        env.draw(delay)
        env.move(ANN.get_move(env))
    env.draw()

def say():
    import os
    os.system('say "gamle ørn, jeg er ferdig  "')


if __name__ == '__main__':
    # Game parameters
    board_size = 3

    # MCTS/RL parameters
    episodes = 10
    simulations = 1000

    #training_batch_size = 100
    ann_save_interval = 5
    eps_decay = 0.99

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.005 # learning rate
    H_dims = [board_size, board_size**2]
    io_dim = board_size * board_size # input and output layer sizes (always equal)
    activation = activation_functions[3]
    optimizer = optimizers[0]
    epochs = 100

    env = HexGame(board_size)
    ANN = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    RL_algorithm(episodes, simulations, env, ANN, eps_decay)
    def play(env = env, ANN = ANN):
        play_game(env,ANN)

    #import pdb; pdb.set_trace()
