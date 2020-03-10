from game import NIMState, LedgeState
from tree import Node
from mcts import MonteCarloTreeSearch
import random
from tqdm import tqdm

def set_starting_player(P):
    assert P == 1 or P == 2 or P == 3, "You can only choose player option 1, 2 or 3"
    if P == 3:
        return random.choice([1,2])
    return P

def print_last_move(iteration, action, player, game_mode):
    msg = ""
    msg += "{}: ".format(iteration+1)
    msg += NIMState.print_move(action.game_state, 3-player, 0) if game_mode == 0 else LedgeState.print_move(0, 3-player, action.game_state)
    msg += "Player "+str(3-player)+" won\n\n"
    return msg

def run_batch(G, M, N, K, B, P, game_mode, verbose):
    wins = 0
    verbose_message = ""
    for i in tqdm(range(G)):
        initial_player = set_starting_player(P)
        action = Node(NIMState(N, K, initial_player) if game_mode == 0 else LedgeState(B, initial_player))
        verbose_message += "Initial state: {}\n".format(action.game_state)
        iteration = 0
        while not action.is_terminal_node():
            player = action.player
            action = MonteCarloTreeSearch(action).best_action(M)
            iteration += 1
            if verbose:
                verbose_message += "{}: ".format(iteration)
                verbose_message += NIMState.print_move(action.prev_action, player, action.game_state) if game_mode == 0 else LedgeState.print_move(action.prev_action, player, action.parent.game_state)
        if initial_player == 3-player:
            wins += 1
        verbose_message += print_last_move(iteration, action, player, game_mode)
    if verbose: print(verbose_message)
    print("Starting player won {}/{} ({}%)".format(wins, G, 100 * wins / G))


if __name__ == '__main__':
    G = 10
    M = 1500
    N = 10
    K = 9
    B = [0, 0, 0, 1, 0, 2, 0, 0, 1, 0]
    P = 1
    game_mode = 1 # (0/1): NIM/Ledge
    verbose = True

    run_batch(G, M, N, K, B, P, game_mode, verbose)
