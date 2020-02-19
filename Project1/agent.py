from actor import Actor
from env import Board
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env, alphaActor, alphaCritic, lam, gamma, criticType, hiddenLayerSizes):
        self.env = env
        self.actor = Actor(alphaActor, lam, gamma)
        self.criticType = criticType
        if criticType == 0: # use criticTable
            from criticTable import CriticTable
            self.critic = CriticTable(alphaCritic, lam, gamma)
        else: # use criticNN
            from criticNN import CriticNN
            state = env.getState()
            inputLayerSize = len(state)
            self.critic = CriticNN(alphaCritic, lam, gamma, hiddenLayerSizes, inputLayerSize)

    # Actor-Critic learning
    def learn(self, runs, eps, epsDecay, verbose = False):
        pegsLeft = []
        iterationNumber = []
        if not verbose: # display progressbar instead
            from tqdm import tqdm
            runList = tqdm(range(runs))
        else:
            runList = range(runs)
        for i in runList: # for each episode
            self.actor.resetEligibilities()
            self.critic.resetEligibilities()
            state, validActions = self.env.reset()
            if self.criticType == 0: # only needed for table critic
                self.critic.createEligibility(state)
                self.critic.createStateValues(state)
            self.actor.createSAPs(state, validActions)
            action = self.actor.findNextAction(state, validActions, eps)
            self.actor.updateEligibility(state, action)
            if len(validActions) == 0: break # do not run episode if initial state gives no valid moves
            while len(validActions) > 0: # while there exist a valid next move
                lastState, state, reinforcement, validActions = self.env.execute(action)
                if self.criticType == 0:
                    self.critic.createEligibility(state)
                    self.critic.createStateValues(state)
                self.actor.createSAPs(state, validActions)
                action = self.actor.findNextAction(state, validActions, eps)
                self.actor.updateEligibility(state, action)
                td_error = self.critic.findTDError(reinforcement, lastState, state)
                if self.criticType == 0:
                    self.critic.updateStateValues()
                else:
                    self.critic.fit(reinforcement, lastState, state, td_error)
                self.critic.updateEligibilities()
                self.actor.updateSAPs(td_error)
                self.actor.decayEligibilities()
            if verbose: # print valuation of each state
                print("ep", i,"  Pegs", self.env.numberOfPegsLeft(), " LastState Value", "%.3f" % self.critic.stateValue(lastState), " eps", "%.3f" % eps)
            pegsLeft.append(self.env.numberOfPegsLeft())
            iterationNumber.append(i)
            eps = eps * epsDecay # decrease exploration
        plt.plot(iterationNumber, pegsLeft) # plot the development for each episode
        plt.show()

    # runs a greedy search through the best states and actions
    def runGreedy(self, delay):
        state, validActions = self.env.reset()
        self.env.draw()
        action = self.actor.findNextAction(state, validActions, 0)
        while len(validActions) > 0: # while there exist a valid next move
            self.env.draw(delay)
            _, state, _, validActions = self.env.execute(action)
            action = self.actor.findNextAction(state, validActions, 0)
        self.env.draw()

if __name__ == '__main__':
    type = 0 # (0/1): triangle/diamond type of board
    size = 5 # size of board
    initial = [(2,1)] # start with hole in (r,c)
    random = 0 # remove random pegs
    env = Board(type, size, initial, random)

    criticValuation = 0 # table/neural net valuation of states. (0/1)
    alphaActor = 0.7 # learning rate actor
    alphaCritic = 0.01 # learning rate critic
    lam = 0.85 # trace-decay
    gamma = 0.9 # discount factor
    hiddenLayerSizes = [5] # structure for hidden layers
    agent = Agent(env, alphaActor, alphaCritic, lam, gamma, criticValuation, hiddenLayerSizes)

    eps = 1
    epsDecay = 0.9
    agent.learn(150, eps, epsDecay, verbose = False)

    # prompt to visualize at the end of learning
    delay = 0.5 # frame delay for visualization
    visualize = input('Do you want to visualize the solution? (y/n): ')
    if visualize == 'y':
        agent.runGreedy(delay)
