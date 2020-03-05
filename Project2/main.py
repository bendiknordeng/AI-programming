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

def get_best_action(simulations_number, state):
    state.turn = 1 # algorithm always plays from perspective of player 1
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
        player = 3 - initial_player
        while not action.is_terminal_node():
            player = 3 - player
            action = get_best_action(M, action.state)
            iteration += 1
            if verbose:
                verbose_message += str(iteration) + ": "
                verbose_message += NIMState.print_move(action, player) if game_mode == 0 else LedgeState.print_move(action, player)
        verbose_message += "Player "+str(player)+" won\n\n"
        if initial_player == player:
            win += 1
    if verbose: print(verbose_message)
    print("Starting player won {}/{} ({}%)".format(win, G, 100 * win / G))


if __name__ == '__main__':
    G = 10
    M = 500
    N = 15
    K = 3
    B = [0, 0, 0, 2, 0, 1]
    P = 1
    game_mode = 0 # (0/1): NIM/Ledge
    verbose = True

    run_batch(G, M, N, K, B, P, game_mode, verbose)
    #[print(get_best_action(M,1,B).game_state) for i in range(50)]
