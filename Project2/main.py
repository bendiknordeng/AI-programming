from game import NIMBoard, LedgeBoard
from tree import Node
from mcts import MonteCarloTreeSearch
import random
import math
from tqdm import tqdm


def set_starting_player(P):
    assert P == 1 or P == 2 or P == 3, "You can only choose player option 1, 2 or 3"
    if P == 3:
        return random.choice([1, 2])
    return P


def print_last_move(iteration, board, game_mode):
    msg = ""
    msg += "{}: ".format(iteration + 1)
    msg += board.print_move(0)
    msg += "Player " + str(board.get_state()[0]) + " won\n\n"
    return msg


def run_batch(G, M, M_decay, N, K, B, P, game_mode, verbose):
    wins = 0
    verbose_message = "\n"
    MCTS = MonteCarloTreeSearch()
    for i in tqdm(range(G)):
        initial_player = set_starting_player(P)
        if game_mode == 0:
            board = NIMBoard(N, K, initial_player)
        else:
            board = LedgeBoard(B, initial_player)
        verbose_message += "Initial state: {}\n".format(board.get_state()[1])
        MCTS.init_tree(board)
        iteration = 0
        simulations = M
        while not board.is_game_over():
            iteration += 1
            action = MCTS.search(board, simulations)
            verbose_message += "{}: ".format(iteration)
            verbose_message += board.print_move(action)
            board.move(action)
            simulations = math.ceil(simulations * M_decay)
        if initial_player == board.player:
            wins += 1
        verbose_message += print_last_move(iteration, board, game_mode)
    if verbose:
        print(verbose_message)
    print("Starting player won {}/{} ({}%)".format(wins, G, 100 * wins / G))


if __name__ == '__main__':
    G = 10
    M = 250
    M_decay = 0
    N = 15
    K = 3
    B = [0, 0, 1, 1, 0, 2]
    P = 1
    game_mode = 1  # (0/1): NIM/Ledge
    verbose = True

    run_batch(G, M, M_decay, N, K, B, P, game_mode, verbose)
