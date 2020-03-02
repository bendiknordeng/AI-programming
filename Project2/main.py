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

def get_best_action(simulations_number, game_mode, game_state, K = 3):
    state = NIMState(game_state, K) if game_mode == 0 else LedgeState(game_state)
    node = Node(state)
    mcts = MonteCarloTreeSearch(node)
    return mcts.best_action(simulations_number)

def run_batch(G, M, N, K, B, P, game_mode, verbose):
    win = 0
    verbose_message = ""
    for i in tqdm(range(G)):
        action = Node(NIMState(N, K) if game_mode == 0 else LedgeState(B))
        verbose_message += "Initial state: {}\n".format(action.game_state)
        iteration = 0
        initial_player = set_starting_player(P)
        player = initial_player
        while True:
            iteration += 1
            action = get_best_action(M, game_mode, action.game_state, K)
            if verbose:
                verbose_message += str(iteration) + ": "
                verbose_message += NIMState.print_move(action, player) if game_mode == 0 else LedgeState.print_move(action, player)
            if action.is_terminal_node():
                break
            player = 3 - player
        verbose_message += "Player "+str(player)+" won\n\n"
        if initial_player == player:
            win += 1
    if verbose: print(verbose_message)
    print("Starting player won {}/{} ({}%)".format(win, G, 100 * win / G))


if __name__ == '__main__':
    G = 10
    M = 500
    N = 10
    K = 3
    B = [1, 0, 0, 2, 0, 1, 0, 0, 0, 1]
    P = 1
    game_mode = 1
    verbose = True

    run_batch(G, M, N, K, B, P, game_mode, verbose)
    #[print(get_best_action(M,1,B).game_state) for i in range(50)]
