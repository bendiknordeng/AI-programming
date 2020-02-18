from actor import Actor
from env import Board
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env, alphaActor, alphaCritic, lam, eps, gamma, criticType, hiddenLayerSizes):
        self.env = env
        self.eps = eps
        self.epsDecay = epsDecay
        self.actor = Actor(alphaActor, lam, gamma)
        self.criticType = criticType
        if criticType == 0: #use criticTable
            from criticTable import CriticTable
            self.critic = CriticTable(alphaCritic, lam, gamma)
        else: #use criticNN
            from criticNN import CriticNN
            state = env.getState()
            inputLayerSize = len(state)
            self.critic = CriticNN(alphaCritic, lam, gamma, hiddenLayerSizes, inputLayerSize)

    def learn(self, runs, verbose = False):
        eps = self.eps
        epsDecay = self.epsDecay
        pegsLeft = []
        iterationNumber = []
        if not verbose:
            runList = tqdm(range(runs))
        else:
            runList = range(runs)
        for i in runList:
            self.actor.resetEligibilities()
            self.critic.resetEligibilities()
            state, validActions = self.env.reset()
            if self.criticType == 0:
                self.critic.createEligibility(state)
                self.critic.createStateValues(state)
            self.actor.createSAPs(state, validActions)
            action = self.actor.findNextAction(state, validActions, eps)
            self.actor.updateEligibility(state, action)
            if len(validActions) == 0: #if state has no valid moves from start, break learning
                break
            while len(validActions) > 0:
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
                self.critic.updateEligibilities() #flyttet utenfor, siden denne skal begge typer critics utføre
                self.actor.updateSAPs(td_error)
                self.actor.decayEligibilities()

            if verbose:
                if self.criticType == 1:
                    print("ep", i,"  Pegs", self.env.numberOfPegsLeft(), " LastState Value", "%.3f" % self.critic.valueState(lastState), " eps", "%.3f" % eps)
                else:
                    print("ep", i,"  Pegs", self.env.numberOfPegsLeft(), " LastState Value", "%.3f" % self.critic.values[lastState], " eps", "%.3f" % eps)
            pegsLeft.append(self.env.numberOfPegsLeft())
            iterationNumber.append(i)

            eps = eps * epsDecay
        plt.plot(iterationNumber, pegsLeft)
        plt.show()

    def runGreedy(self, delay):
        self.env.reset()
        self.env.draw()
        reinforcement = 0
        state = self.env.getState()
        validActions = self.env.generateActions()
        action = self.actor.findNextAction(state, validActions, 0)
        while len(validActions) > 0:
            self.env.draw(delay)
            self.env.execute(action)
            reinforcement = self.env.reinforcement()
            state = self.env.getState()
            self.actor.createSAPs(state, self.env.generateActions())
            validActions = self.env.generateActions()
            action = self.actor.findNextAction(state, validActions, 0)
        self.env.draw()


if __name__ == '__main__':
    type = 0
    size = 6
    initial = [(2,1)] # start with hole in (r,c)
    random = 0 # remove random pegs
    env = Board(type, size, initial, random)
    #env.draw()
    delay = 0.5 # for visualization

    criticValuation = 1 # table/neural net valuation of states. (0/1)
    alphaActor = 0.7
    alphaCritic = 0.01
    lam = 0.85
    gamma = 0.9
    eps = 1
    epsDecay = 0.9
    hiddenLayerSizes = [5]
    agent = Agent(env, alphaActor, alphaCritic, lam, eps, gamma, criticValuation, hiddenLayerSizes)

    agent.learn(700, verbose = True)
    # visualize = input('Do you want to visualize the solution? (y/n): ')
    # if visualize == 'y':
    #     agent.runGreedy(delay)
