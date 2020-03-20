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
    for i in tqdm(range(games)):
        env.reset()
        MCTS.init_tree()
        M = simulations
        while not env.is_game_over():
            action, D = MCTS.search(env, M)
            cases.append((env.flat_state,D))
            env.move(action)
            M = math.ceil(M*0.5)
        ANN.fit(random.sample(cases, math.ceil(len(cases)/math.floor(math.sqrt(i+1)))))
        #if (i+1) % save_interval == 0:
        #    ANN.model.save_weights(model_path.format(level=i+1))
        MCTS.eps *= eps_decay

def play(env, ANN, delay,verbose):
    env.reset()
    while not env.is_game_over():
        env.draw(delay)
        if verbose: print(ANN.forward(env.flat_state))
        env.move(ANN.get_move(env))
    env.draw()


if __name__ == '__main__':
    # Game parameters
    board_size = 3

    # MCTS/RL parameters
    episodes = 50
    simulations = 1000

    #training_batch_size = 100
    ann_save_interval = 5
    eps_decay = 1

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.01 # learning rate
    H_dims = [board_size, board_size**2]
    io_dim = board_size * board_size # input and output layer sizes (always equal)
    activation = activation_functions[3]
    optimizer = optimizers[0]
    epochs = 10

    env = HexGame(board_size)
    ANN = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    RL_algorithm(episodes, simulations, env, ANN, eps_decay)

    import pdb; pdb.set_trace()

    """
    def calc_time(time_random = np.asarray(time_rnd), time_ANN = np.asarray(time_ANN)):
        sum_rnd = np.sum(time_random, axis=0)
        sum_ANN = np.sum(time_ANN, axis=0)
        avg_rnd = sum_rnd / len(time_rnd)
        avg_ANN = sum_ANN / len(time_ANN)
        print("avg random",avg_rnd)
        print("avg ann",avg_ANN)
        print("sum_time_ann_rollout / sum_time_random_rollout",avg_ANN/avg_rnd)
        print("sum_rnd",sum_rnd)
        print("sum_ANN",sum_ANN)
    """
