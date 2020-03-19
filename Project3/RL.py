from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
import random
from tqdm import tqdm
import math


def RL_algorithm(games, simulations, env, ANN, eps_decay, training_batch_size):
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
        ANN.fit(random.sample(cases, min(len(cases),training_batch_size)))
        #if (i+1) % save_interval == 0:
        #    ANN.model.save_weights(model_path.format(level=i+1))
        MCTS.eps *= eps_decay

if __name__ == '__main__':
    # Game parameters
    board_size = 3

    # MCTS/RL parameters
    episodes = 50
    simulations = 500
    training_batch_size = 100
    ann_save_interval = 5
    eps_decay = 0.95

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.01 # learning rate
    H_dims = [10,10,10]
    io_dim = board_size * board_size # input and output layer sizes (always equal)
    activation = activation_functions[3]
    optimizer = optimizers[0]
    epochs = 10

    env = HexGame(board_size)
    ANN = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    RL_algorithm(episodes, simulations, env, ANN, eps_decay, training_batch_size)

    import pdb; pdb.set_trace()
