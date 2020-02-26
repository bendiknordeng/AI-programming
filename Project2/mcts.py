from game import NIM, Ledge
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class MCTS:
    def __init__(self, B, M, env, eps, eps_decay, c):
        self.G = G
        self.M = M
        self.env = env
        self.eps = eps
        self.eps_decay = eps_decay
        self.c = c

    def tree_search(self, display):
        winner = []
        iteration_number = []
        current = self.env.root
        for i in tqdm(range(self.G)):
            while not self.env.final_state(current.state):
                current = self.__find_best_leaf(current)
                for j in range(self.M):
                    self.__rollout(current)
                if self.env.final_state(current.state):
                    if current.turn:
                        winner.append(1)
                    else:
                        winner.append(2)
                    current = self.env.root
                    break
            iteration_number.append(i)
            self.eps = self.eps * self.eps_decay
        if display:
            plt.plot(iteration_number, winner) # plot the development for each episode
            plt.show()

        print("Player one wins: {}\nPlayer two wins: {}".format(
            winner.count(1), winner.count(2)))

    def run_greedy(self):
        current = self.env.root
        print("Initial state: {}".format(current.state))
        while not self.env.final_state(current.state):
            current = self.__greedy_choice(current)
            self.env.print_move(current)
        print('Player {} won after {} moves.\n'.format(
            1 if current.parent.turn else 2, current.count_parents()))

    def display_tree(self):
        current = self.env.root
        print(current)
        while current.children:
            current = current.get_best_child()
            print(current)

    def __find_best_leaf(self, node):
        current = node
        while current.children:
            current = self.__greedy_choice(current)
        return current

    def __greedy_choice(self, current):
        return current.get_best_child()

    def __rollout(self, node):
        current = node
        while not self.env.final_state(current.state):
            current.visits += 1
            if not current.children:
                self.env.generate_child_states(current)
            if random.random() > self.eps:
                current = self.__greedy_choice(current)
            else:
                current = random.choice(current.children)
        current.visits += 1
        self.__backprop(current)

    def __backprop(self, node):
        current = node
        reinforcement = self.env.get_reinforcement(current)
        while current.parent != self.env.root:
            current.update_values(reinforcement, self.c)
            current = current.parent


if __name__ == '__main__':
    G = 100  # number of games in batch
    M = 50  # number of rollouts per game move
    P = 1  # (1/2/3): Player 1 starts/Player 2 starts/Random player starts
    c = 1  # exploration constant
    eps = 1
    eps_decay = 0.9
    game_mode = 0  # (0/1): NIM/Ledge

    N = 10  # Inittial pile for NIM
    K = 3  # Max pieces for each action in NIM
    B = [1, 1, 1, 0, 0, 2, 0]  # board for ledge

    if game_mode == 0:
        env = NIM(P, N, K)
    else:
        env = Ledge(P, B)

    display = True
    mcts = MCTS(G, M, env, eps, eps_decay, c)
    mcts.tree_search(display)
    mcts.display_tree()
    print()
    mcts.run_greedy()
