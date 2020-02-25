from game import NIM, Ledge
from tqdm import tqdm
import random
import math


class MCTS:
    def __init__(self, G, M, env, eps, eps_decay, c):
        self.G = G
        self.M = M
        self.env = env
        self.Q = {}  # state action values for tree edges
        self.u = {}  # exploration bonus
        self.eps = eps
        self.eps_decay = eps_decay
        self.c = c

    def tree_search(self):
        current = self.env.root
        for i in tqdm(range(self.G)):
            for j in range(self.M):
                self.__rollout(current)
            current = self.__greedy_choice(current)
            if self.env.final_state(current.state):
                break
            self.eps = self.eps * self.eps_decay

    def run_greedy(self):
        current = self.env.root
        while not self.env.final_state(current):
            current = self.__greedy_choice(current)
            print(current)

    def __greedy_choice(self, current):
        saps = []
        for action in current.actions:
            if not self.Q.get((current,action)) == None:
                saps.append((current,action))
        choices = {}
        if current.turn:
            for sap in saps:
                choices[sap] = self.Q.get(sap) + self.u.get(sap)
            return max(choices, key=choices.get)[1].child
        else:
            for sap in saps:
                choices[sap] = self.Q.get(sap) - self.u.get(sap)
            return min(choices, key=choices.get)[1].child

    def __rollout(self, node):
        current = node
        while not self.env.final_state(current.state):
            if not current.children:
                env.generate_child_states(current)
            #if random.random() > self.eps:
            #    current = self.__greedy_choice(current)
            #else:
            current = random.choice(current.children)
        self.__back_propagate(current)

    def __back_propagate(self, node):
        current = node
        current.visits += 1
        while current.parent:
            current.parent.visits += 1
            if current.parent: # if not root node
                current.prev_action.visits += 1
                self.u[(current.parent, current.prev_action)] = self.c * \
                math.sqrt(math.log(current.parent.visits) /
                          (1 + current.prev_action.visits))
                reinforcement = self.env.get_reinforcement(current)
                current.prev_action.update_value(reinforcement)
                self.Q[(current.parent, current.prev_action)
                       ] = current.prev_action.value / current.prev_action.visits
            current = current.parent


if __name__ == '__main__':
    G = 10  # number of games in batch
    M = 10  # number of rollouts per game move
    P = 3  # (1/2/3): Player 1 starts/Player 2 starts/Random player starts
    c = 1  # exploration constant
    eps = 1
    eps_decay = 0.9
    game_mode = 0  # (0/1): NIM/Ledge

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
