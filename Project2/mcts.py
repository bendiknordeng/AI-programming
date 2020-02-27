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
        run_list = range(self.G) if verbose else tqdm(range(self.G))
        for i in run_list:
            self.env = NIM(P, N, K) if game_mode == 0 else Ledge(P, B)
            current_root = self.env.root
            if verbose: print("Initial state: {}".format(current_root.state))
            while not self.env.final_state(current_root.state):
                for j in range(self.M):
                    leaf = self.__find_best_leaf(current_root) # select
                    self.env.generate_child_states(leaf) # expand
                    rollout_node = random.choice(leaf.children) if leaf.children else leaf
                    reward = self.__rollout(rollout_node) # simulation
                    self.__backprop(rollout_node, reward) # backpropegation
                current_root = current_root.get_best_child(leaf_search = False, verbose = True)
                if verbose: self.env.print_move(current_root)
            if current_root.parent.turn == self.env.root.turn:
                win += 1
            if verbose:
                print("Player {} wins after {} moves\n".format(
                    1 if leaf.parent.turn else 2, leaf.count_parents()))

        print("Starting player wins: {}/{} ({:.2f}%)".format(
            win, self.G, win * 100 / self.G))

    def __find_best_leaf(self, node):
        current = node
        while current.children:
            current = current.get_best_child(leaf_search = True, verbose = False)
        return current

    def __rollout(self, node):
        state = node.state
        turn = node.turn
        while not self.env.final_state(state):
            action = random.choice(self.env.generate_valid_actions(state))
            state = self.env.next_state(state, action)
            turn = not turn
        return self.env.get_reinforcement(not turn)

    def __backprop(self, node, reinforcement):
        current = node
        current.visits += 1
        while current != self.env.root:
            current.parent.visits += 1
            current.update_values(reinforcement, self.c)
            current = current.parent

if __name__ == '__main__':
    G = 50  # number of games in batch
    M = 500  # number of rollouts per game move
    P = 2  # (1/2/3): Player 1 starts/Player 2 starts/Random player startsudfar
    c = 1  # exploration constant
    N = 10  # Inittial pile for NIM
    K = 3  # Max pieces for each action in NIM
    B = [1, 0, 0, 2]  # board for ledge
    game_mode = 0  # (0/1): NIM/Ledge

    mcts = MCTS(G, M, c)

    verbose = False
    mcts.tree_search(game_mode, P, N, K, B, False)
