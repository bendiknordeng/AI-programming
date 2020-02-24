from game import NIM, Ledge
from tqdm import tqdm
import random
import math


class MCTS:
    def __init__(self, G, M, env, eps, epsDecay, c):
        self.G = G
        self.M = M
        self.env = env
        self.Q = {}  # state action values for tree edges
        self.u = {}  # exploration bonus
        self.eps = eps
        self.epsDecay = epsDecay
        self.c = c

    def learn(self):
        current = self.env.root
        for i in tqdm(range(self.G)):
            for j in range(self.M):
                self.__rollout(current)
            current = self.__greedyChoice(current)
            if self.env.finalState(current.state):
                break
            self.eps = self.eps * self.epsDecay

    def runGreedy(self):
        current = self.env.root
        while not self.env.finalState(current):
            current = self.__greedyChoice(current)

    def __greedyChoice(self, current):
        saps = []
        for action in current.actions:
            if not self.Q.get((current,action)) == None:
                saps.append((current,action))
        choices = {}
        if current.turn:
            for sap in saps:
                choices[sap] = self.Q.get(sap) + self.u.get(sap)
            return max(choices, key=choices.get)[0]
        else:
            for sap in saps:
                choices[sap] = self.Q.get(sap) - self.u.get(sap)
            return min(choices, key=choices.get)[0]

    def __rollout(self, node):
        current = node
        while not self.env.finalState(current.state):
            if not current.children:
                env.generateChildStates(current)
            if random.random() > self.eps:
                current = self.__greedyChoice(current)
            else:
                current = random.choice(current.children)
        self.__backPropagate(current)

    def __backPropagate(self, node):
        current = node
        current.visits += 1
        while current.parent:
            current.parent.visits += 1
            if current.parent: # if not root node
                current.prevAction.visits += 1
                self.u[(current.parent, current.prevAction)] = self.c * \
                math.sqrt(math.log(current.parent.visits) /
                          (1 + current.prevAction.visits))
                reinforcement = self.env.getReinforcement(current)
                current.prevAction.updateValue(reinforcement)
                self.Q[(current.parent, current.prevAction)
                       ] = current.prevAction.value / current.prevAction.visits
            current = current.parent


if __name__ == '__main__':
    G = 10  # number of games in batch
    M = 10  # number of rollouts per game move
    P = 3  # (1/2/3): Player 1 starts/Player 2 starts/Random player starts
    c = 1  # exploration constant
    eps = 1
    epsDecay = 0.9
    gameMode = 0  # (0/1): NIM/Ledge

    N = 20  # Inittial pile for NIM
    K = 5  # Max pieces for each action in NIM
    B = [1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1]  # board for ledge

    if gameMode == 0:
        env = NIM(P, N, K)
    else:
        env = Ledge(P, B)

    mcts = MCTS(G, M, env, eps, epsDecay, c)
    mcts.learn()
    #mcts.runGreedy()
