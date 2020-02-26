from game import NIM, Ledge
import random
from tqdm import tqdm

class MCTS:
    def __init__(self, B, M, c):
        self.G = G
        self.M = M
        self.c = c

    def tree_search(self, game_mode, P, N, K, B, verbose):
        win = 0
        for i in range(self.G):
            if game_mode == 0:
                self.env = NIM(P, N, K)
            else:
                self.env = Ledge(P, B)
            current = self.env.root
            if verbose: print("Initial state: {}".format(current.state))

            while not self.env.final_state(current.state):
                for j in range(self.M):
                    self.__rollout(current)
                current, _ = current.get_best_child()
                if verbose: self.env.print_move(current)

            if current.parent.turn == self.env.root.turn:
                win += 1
            if verbose:
                print("Player {} wins after {} moves\n".format(
                    1 if current.parent.turn else 2, current.count_parents()))

        print("Starting player wins: {}/{} ({}%)".format(
            win, self.G, win * 100 / self.G))


    def __rollout(self, node):
        current = node
        while not self.env.final_state(current.state):
            current.visits += 1
            if not current.children:
                self.env.generate_child_states(current)
            current = random.choice(current.children)
        current.visits += 1  # add visit to final node
        self.__backprop(current)

    def __backprop(self, node):
        current = node
        reinforcement = self.env.get_reinforcement(current)
        while True:
            current.update_values(reinforcement, self.c)
            current = current.parent
            if current == self.env.root:
                break


if __name__ == '__main__':
    G = 10  # number of games in batch
    M = 500  # number of rollouts per game move
    P = 1  # (1/2/3): Player 1 starts/Player 2 starts/Random player startsudfar
    c = 1  # exploration constant
    N = 10  # Inittial pile for NIM
    K = 3  # Max pieces for each action in NIM
    B = [1, 1, 0, 0, 2, 1]  # board for ledge
    game_mode = 0  # (0/1): NIM/Ledge

    mcts = MCTS(G, M, c)

    verbose = True
    mcts.tree_search(game_mode, P, N, K, B, verbose)
