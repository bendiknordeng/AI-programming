from game import NIMState, LedgeState
from tree import Node
from mcts import MonteCarloTreeSearch
import random
from tqdm import tqdm


def starting_player(P):
    assert P == 1 or P == 2 or P == 3, "You can only choose between player option 1, 2 or 3"
    if P == 3:
        return random.choice([1,2])
    return P

if __name__ == '__main__':
    G = 10
    M = 500
    N = 10
    K = 3
    B = [0, 0, 0, 1, 0, 2, 0, 0, 1, 0]
    P = starting_player(1)
    game_mode = 0

    winner = []
    for i in tqdm(range(G)):
        state = NIMState(N, K, P) if game_mode == 0 else LedgeState(B, P)
        current = Node(state)
        mcts = MonteCarloTreeSearch(current)
        action = mcts.best_action(M)
        if state.is_game_over:
            winner.append(state.game_result)

    print(winner)
