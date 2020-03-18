from game import NIMBoard, LedgeBoard
from tree import Node
from mcts import MonteCarloTreeSearch
import config as cfg
import random
import math
from tqdm import tqdm


def set_starting_player(P):
    assert P == 1 or P == 2 or P == 3, "You can only choose player option 1, 2 or 3"
    if P == 3:
        return random.choice([1, 2])
    return P


def print_last_move(iteration, env):
    msg = "{}: {}".format(iteration, env.print_move(0))
    msg += "Player " + str(env.player) + " won\n\n"
    return msg


def run_batch(G, M, M_decay, N, K, B, P, game_mode, verbose):
    wins = 0
    verbose_message = "\n"
    MCTS = MonteCarloTreeSearch()
    for i in tqdm(range(G)):
        starting_player = set_starting_player(P)
        if game_mode == 0:
            env = NIMBoard(N, K, starting_player)
        else:
            env = LedgeBoard(B, starting_player)
        verbose_message += "Initial state: {}\n".format(env.get_state()[1])
        MCTS.init_tree(env)
        iteration = 1
        simulations = M
        while not env.is_game_over():
            action = MCTS.search(env, simulations) # find best move
            verbose_message += "{}: {}".format(iteration, env.print_move(action))
            env.move(action)
            iteration += 1
            simulations = math.ceil(simulations * M_decay) # (optional) speed up for mcts
        if starting_player == env.player:
            wins += 1
        verbose_message += print_last_move(iteration, env)
    if verbose:
        print(verbose_message)
    print("Starting player won {}/{} ({}%)".format(wins, G, 100 * wins / G))

if __name__ == '__main__':
    G = cfg.G
    M = cfg.M
    M_decay = cfg.M_decay
    N = cfg.N
    K = cfg.K
    B = cfg.B
    P = cfg.P
    game_mode = cfg.game_mode  # (0/1): NIM/Ledge
    verbose = cfg.verbose

    run_batch(G, M, M_decay, N, K, B, P, game_mode, verbose)
