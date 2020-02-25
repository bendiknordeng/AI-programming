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

    def tree_search(self):
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
        plt.plot(iteration_number, winner) # plot the development for each episode
        plt.show(block = False)
        plt.pause(1)
        plt.close()

        print("Player one wins: {}\nPlayer two wins: {}".format(
            winner.count(1), winner.count(2)))

    def run_greedy(self):
        current = self.env.root
        moves = 0
        print("Initial state: {}".format(current.state))
        while not self.env.final_state(current.state):
            moves += 1
            current = self.__greedy_choice(current)
            self.env.print_move(current)
        print('Player {} won after {} moves.\n'.format(
            1 if current.parent.turn else 2, moves))

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
        while current.parent != self.env.root:
            reinforcement = self.env.get_reinforcement(current)
            current.update_values(reinforcement, self.c)
            current = current.parent


if __name__ == '__main__':
    G = 100  # number of games in batch
    M = 10  # number of rollouts per game move
    P = 1  # (1/2/3): Player 1 starts/Player 2 starts/Random player starts
    c = 1  # exploration constant
    eps = 1
    eps_decay = 0.9
    game_mode = 0  # (0/1): NIM/Ledge

    N = 20  # Inittial pile for NIM
    K = 3  # Max pieces for each action in NIM
    B = [1, 0, 0, 1, 0, 0, 0, 2, 0, 0]  # board for ledge

    if game_mode == 0:
        env = NIM(P, N, K)
    else:
        env = Ledge(P, B)

    mcts = MCTS(G, M, env, eps, eps_decay, c)
    mcts.tree_search()
    mcts.run_greedy()
