from actor import Actor
from env import Board
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env, alphaActor, alphaCritic, lam, gamma, criticType, hiddenLayerSizes):
        self.__env = env
        self.__actor = Actor(alphaActor, lam, gamma)
        self.__criticType = criticType
        if self.__criticType == 0: # use criticTable
            from criticTable import CriticTable
            self.__critic = CriticTable(alphaCritic, lam, gamma)
        else: # use criticNN
            from criticNN import CriticNN
            state = env.getState()
            inputLayerSize = len(state)
            self.__critic = CriticNN(alphaCritic, lam, gamma, hiddenLayerSizes, inputLayerSize)

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
            self.__actor.resetEligibilities()
            self.__critic.resetEligibilities()
            state, validActions = self.__env.reset()
            if self.__criticType == 0: # only needed for table critic
                self.__critic.createEligibility(state)
                self.__critic.createStateValues(state)
            self.__actor.createSAPs(state, validActions)
            action = self.__actor.findNextAction(state, validActions, eps)
            self.__actor.updateEligibility(state, action)
            if len(validActions) == 0: break # do not run episode if initial state gives no valid moves
            while len(validActions) > 0: # while there exist a valid next move
                lastState, state, reinforcement, validActions = self.__env.execute(action)
                if self.__criticType == 0:
                    self.__critic.createEligibility(state)
                    self.__critic.createStateValues(state)
                self.__actor.createSAPs(state, validActions)
                action = self.__actor.findNextAction(state, validActions, eps)
                self.__actor.updateEligibility(state, action)
                td_error = self.__critic.findTDError(reinforcement, lastState, state)
                if self.__criticType == 0:
                    self.__critic.updateStateValues()
                else:
                    self.__critic.fit(reinforcement, lastState, state, td_error)
                self.__critic.updateEligibilities()
                self.__actor.updateSAPs(td_error)
                self.__actor.decayEligibilities()
            if verbose: # print valuation of each state
                print("ep", i,"  Pegs", self.__env.numberOfPegsLeft(), " LastState Value", "%.3f" % self.__critic.stateValue(lastState), " eps", "%.3f" % eps)
            pegsLeft.append(self.__env.numberOfPegsLeft())
            iterationNumber.append(i)
            eps = eps * epsDecay # decrease exploration
        plt.plot(iterationNumber, pegsLeft) # plot the development for each episode
        plt.show()

    # runs a greedy search through the best states and actions
    def runGreedy(self, animation_delay):
        state, validActions = self.__env.reset()
        self.__env.draw()
        action = self.__actor.findNextAction(state, validActions, 0)
        while len(validActions) > 0: # while there exist a valid next move
            self.__env.draw(animation_delay)
            _, state, _, validActions = self.__env.execute(action)
            self.__actor.createSAPs(state, validActions) # if game is not won, greedy run may encounter new states.
            action = self.__actor.findNextAction(state, validActions, 0)
        self.__env.draw()

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
    agent.learn(150, eps, epsDecay, verbose = True)

    # prompt to visualize at the end of learning
    animation_delay = 0.5 # frame delay for visualization
    visualize = input('Do you want to visualize the solution? (y/n): ')
    if visualize == 'y':
        agent.runGreedy(animation_delay)
