from game import NIM, Ledge
import random
from tqdm import tqdm
import matplotlib as plt

class MCTS:
    def __init__(self, B, M, env, eps, eps_decay, c):
        self.G = G
        self.M = M
        self.env = env
        self.eps = eps
        self.eps_decay = eps_decay
        self.c = c

    def tree_search(self):
        player_wins = [0,0]
        current = self.env.root
        for i in tqdm(range(self.G)):
            while not self.env.final_state(current.state):
                current = self.__find_best_leaf(current)
                for j in range(self.M):
                    self.__rollout(current)
                if self.env.final_state(current.state):
                    if current.turn:
                        player_wins[0] += 1
                    else:
                        player_wins[1] += 1
                    current = self.env.root
                    break
            self.eps = self.eps * self.eps_decay
        print("Player one wins: {}\nPlayer two wins: {}".format(player_wins[0],player_wins[1]))

    def run_greedy(self):
        current = self.env.root
        moves = 0
        print("Initial state: {}".format(current.state))
        print("Player {} starts".format(1 if current.turn else 2))
        while not self.env.final_state(current.state):
            moves+=1
            current = self.__greedy_choice(current)
            print("Move: {}, Q-value: {}, u-value: {}".format(current.count_parents(),current.Q, current.u))

        print('Player {} won, after {} moves.\n'.format(1 if current.turn else 2,moves))


    def __find_best_leaf(self, node):
        current = node
        while current.children:
            current = self.__greedy_choice(current)
        return current

    def __greedy_choice(self, current):
        best_node = random.choice(current.children)
        best_value = 0
        for node in current.children:
            if node.turn:
                if node.Q + node.u >= best_value:
                    best_node = node
                else:
                    if node.Q - node.u <= best_value:
                        best_node = node
        return best_node

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
    M = 10 # number of rollouts per game move
    P = 1  # (1/2/3): Player 1 starts/Player 2 starts/Random player starts
    c = 1  # exploration constant
    eps = 1
    eps_decay = 0.99
    game_mode = 0 # (0/1): NIM/Ledge

    N = 20  # Inittial pile for NIM
    K = 5  # Max pieces for each action in NIM
    B = [1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1]  # board for ledge

    if game_mode == 0:
        env = NIM(P, N, K)
    else:
        env = Ledge(P, B)

    mcts = MCTS(G, M, env, eps, eps_decay, c)
    mcts.tree_search()
    mcts.run_greedy()
