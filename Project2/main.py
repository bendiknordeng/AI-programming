from game import NIMBoard, LedgeBoard
from tree import Node
from mcts import MonteCarloTreeSearch
import random
from tqdm import tqdm

def set_starting_player(P):
    assert P == 1 or P == 2 or P == 3, "You can only choose player option 1, 2 or 3"
    if P == 3:
        return random.choice([1,2])
    return P

def print_last_move(iteration, action, board, game_mode):
    msg = ""
    msg += "{}: ".format(iteration+1)
    msg += NIMBoard.print_move(board.get_state()[1],board) if game_mode == 0 else LedgeBoard.print_move(0, board)
    msg += "Player "+str(board.get_state()[0])+" won\n\n"
    return msg

def run_batch(G, M, N, K, B, P, game_mode, verbose):
    wins = 0
    verbose_message = "\n"
    MCTS = MonteCarloTreeSearch()
    for i in tqdm(range(G)):
        initial_player = set_starting_player(P)
        board = NIMBoard(N, K, initial_player) if game_mode == 0 else LedgeBoard(B, initial_player)
        _, state = board.get_state()
        verbose_message += "Initial state: {}\n".format(state)
        iteration = 0
        while not board.is_game_over():
            iteration += 1
            MCTS.init_tree(board)
            action = MCTS.search(board, M)
            if verbose:
                verbose_message += "{}: ".format(iteration)
                verbose_message += NIMBoard.print_move(action, board) if game_mode == 0 else LedgeBoard.print_move(action, board)
            board.move(action)
        if (initial_player == 1 and board.player1_won()) or (initial_player == 2 and not board.player1_won()):
            wins += 1
        verbose_message += print_last_move(iteration, action, board, game_mode)
    if verbose: print(verbose_message)
    print("Starting player won {}/{} ({}%)".format(wins, G, 100 * wins / G))


if __name__ == '__main__':
    G = 10
    M = 500
    N = 25
    K = 3
    B = [1, 0, 0, 2, 0, 1, 0, 1]
    P = 1
    game_mode = 0 # (0/1): NIM/Ledge
    verbose = True

    run_batch(G, M, N, K, B, P, game_mode, verbose)
