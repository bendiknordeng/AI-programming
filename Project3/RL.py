from game import HexGame
from mcts import MonteCarloTreeSearch
from ANN import ANN
import random
from tqdm import tqdm


def RL_algorithm(games, simulations, env, ANN, eps_decay, training_batch_size):
    cases = [[],[]]
    for i in tqdm(range(games)):
        env.reset()
        MCTS = MonteCarloTreeSearch(ANN)
        while not env.is_game_over():
            action, D = MCTS.search(env, simulations)
            cases[0].append(env.flat_state)
            cases[1].append(D)
            env.move(action)
        ANN.fit(random.sample(cases, min(len(cases[0]),training_batch_size)))
        #if (i+1) % save_interval == 0:
        #    ANN.model.save_weights(model_path.format(level=i+1))
        MCTS.eps *= eps_decay

if __name__ == '__main__':
    # Game parameters
    board_size = 3

    # MCTS/RL parameters
    episodes = 20
    simulations = 500
    training_batch_size = 10
    ann_save_interval = 5
    eps_decay = 0.95

    # ANN parameters
    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001 # learning rate
    H_dims = [10,10,10]
    io_dim = board_size * board_size # input and output layer sizes (always equal)
    activation = activation_functions[3]
    optimizer = optimizers[0]
    epochs = 10

    env = HexGame(board_size)
    ANN = ANN(io_dim, H_dims, alpha, optimizer, activation, epochs)
    RL_algorithm(episodes, simulations, env, ANN, eps_decay, training_batch_size)
    import pdb; pdb.set_trace()
