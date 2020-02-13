from actor import Actor
from criticTable import CriticTable
from criticNN import CriticNN
from env import Board
import numpy as np
import time
from progressbar import ProgressBar
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env, alphaActor, alphaCritic, eps, gamma, criticType, nodesInLayers):
        self.env = env
        self.eps = eps
        self.epsDecay = epsDecay
        self.actor = Actor(alphaActor, lambdod, gamma)
        if criticType == 0: #use criticTable
            self.critic = CriticTable(alphaCritic, lambdod, gamma, inputDim, nodesInLayers)
        else: #use criticNN
            state = env.getState()
            inputDim = len((np.array([int(bin) for bin in state])))
            self.critic = CriticNN(alphaCritic, lambdod, gamma, inputDim, nodesInLayers)
            self.env.execute(((2,0),(0,0)))
            nextState = env.getState()
            self.critic.fitNeuralNet(self.env.reinforcement(), state, nextState)

    def learn(self, runs):
        eps = self.eps
        epsDecay = self.epsDecay
        pegsLeft = []
        iterationNumber = []
        iteration = 0
        start_time = time.time()
        pbar = ProgressBar()
        for i in pbar(range(runs)):
            iteration += 1
            eps = eps*epsDecay

            self.env.reset()
            reinforcement = 0
            #initialize new state values and (s,a)-pairs for start (s,a)
            state = self.env.getState()
            self.critic.createEligibility(state)
            self.critic.createStateValues(state)
            validActions = self.env.generateActions()
            self.actor.createSAPs(state, validActions)
            self.actor.createEligibilities(state, validActions)
            action = self.actor.findNextAction(state, eps)
            while len(validActions) > 0 :
                #make move
                #self.env.draw()
                self.env.execute(action)
                lastAction = action
                lastState = state
                state = self.env.getState()
                validActions = self.env.generateActions()
                #initialize new state values and (s,a)-pairs underway
                self.critic.createEligibility(state)
                self.critic.createStateValues(state)
                self.actor.createSAPs(state, validActions)
                self.actor.createEligibilities(state, validActions)
                reinforcement = self.env.reinforcement()
                action = self.actor.findNextAction(state, eps)
                if not action == -1:
                    self.actor.updateNextEligibility(state, action)
                surprise = self.critic.findTDError(reinforcement, lastState, state)
                self.critic.updateCurrentEligibility(lastState)
                self.critic.updateStateValues()
                self.critic.updateEligibilities()
                self.actor.updateSAPs(surprise)
                self.actor.updateEligibilities()
            pegsLeft.append(self.env.numberOfPegsLeft())
            iterationNumber.append(i)

        timeSpend = time.time()- start_time
        print("time spend", timeSpend)
        plt.plot(iterationNumber, pegsLeft);
        plt.show()

    def runGreedy(self):
        start_time = time.time()
        self.env.reset()
        self.env.draw()
        reinforcement = 0
        state = self.env.getState()
        action = self.actor.findNextAction(state, 0)
        while len(self.env.generateActions()) > 0:
            self.env.draw(0.5)
            self.env.execute(action)
            reinforcement = self.env.reinforcement()
            state = self.env.getState()
            self.actor.createSAPs(state, self.env.generateActions())
            action = self.actor.findNextAction(state, 0)
        self.env.draw()

if __name__ == '__main__':
    type = 0
    size = 5
    initial = [(0,0)] # start with hole in (r,c)
    random = 0 # remove random pegs
    env = Board(type, size, initial, random)

    alpha = 0.2
    lambdod = 0.9  #lambda
    gamma = 0.95
    eps = 1
    epsDecay = 0.9
    criticValuation = 1 #neural net valuation of states.
    nodesInLayers = [5,5,1]
    agent = Agent(env, alpha, alpha, eps, gamma, criticValuation, nodesInLayers)

    #agent.learn(300)
    #agent.runGreedy()
