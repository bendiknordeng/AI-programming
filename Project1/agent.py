from actor import Actor
from criticTable import CriticTable
from criticNN import CriticNN
from env import Board
from progressbar import ProgressBar
import numpy as np
import time
import matplotlib.pyplot as plt



class Agent:
    def __init__(self, env, alphaActor, alphaCritic, lam, eps, gamma, criticType, nodesInLayers):
        self.env = env
        self.eps = eps
        self.epsDecay = epsDecay
        self.actor = Actor(alphaActor, lam, gamma)
        self.criticType = criticType
        if criticType == 0: #use criticTable
            self.critic = CriticTable(alphaCritic, lam, gamma)
        else: #use criticNN
            state = env.getState()
            inputDim = len((np.array([int(bin) for bin in state])))
            self.critic = CriticNN(alphaCritic, lam, gamma, inputDim, nodesInLayers)

    def learn(self, runs):
        eps = self.eps
        epsDecay = self.epsDecay
        pegsLeft = []
        iterationNumber = []
        iteration = 0
        start_time = time.time()
        #pbar = ProgressBar()
        for i in range(runs):#pbar(range(runs)):
            iteration += 1
            self.env.reset()
            self.actor.resetEligibilities()
            self.critic.resetEligibilities()
            # initialize new state values and (s,a)-pairs for start (s,a)
            state = self.env.getState()
            if self.criticType == 0:
                self.critic.createEligibility(state)
                self.critic.createStateValues(state)

            validActions = self.env.generateActions()
            self.actor.createSAPs(state, validActions)
            self.actor.createEligibilities(state, validActions)
            action = self.actor.findNextAction(state, validActions, eps)
            #self.env.draw()
            while len(validActions) > 0:
                lastState = state # save current state before new action
                self.env.execute(action)
                state = self.env.getState()
                validActions = self.env.generateActions()

                if self.criticType == 0:
                    self.critic.createEligibility(state)
                    self.critic.createStateValues(state)

                self.actor.createSAPs(state, validActions)
                self.actor.createEligibilities(state, validActions)

                reinforcement = self.env.reinforcement()
                action = self.actor.findNextAction(state, validActions, eps)

                self.actor.updateCurrentEligibility(state, action)
                td_error = self.critic.findTDError(reinforcement, lastState, state)

                if self.criticType == 0:
                    self.critic.updateCurrentEligibility(lastState)
                    self.critic.updateStateValues()
                else:
                    self.critic.fit(reinforcement, lastState, state, td_error)
                self.critic.updateEligibilities() #flyttet utenfor, siden denne skal begge typer critics utføre

                self.actor.updateSAPs(td_error)
                self.actor.updateEligibilities()

            print("ep", i,"  Pegs", self.env.numberOfPegsLeft(), " LastState Value", "%.3f" % self.critic.modelPred(lastState), " eps", "%.3f" % eps)
            pegsLeft.append(self.env.numberOfPegsLeft())
            iterationNumber.append(i)
            #if i > 250:
            #    eps=1
            #else:
            eps = eps * epsDecay
        time_spent = time.time() - start_time
        print("Time spent", time_spent)
        plt.plot(iterationNumber, pegsLeft)
        plt.show()

    def runGreedy(self, visualizeSolution, delay):
        start_time = time.time()
        self.env.reset()
        if visualizeSolution:
            self.env.draw()
        reinforcement = 0
        state = self.env.getState()
        validActions = self.env.generateActions()
        action = self.actor.findNextAction(state, validActions, 0)
        while len(validActions) > 0:
            if visualizeSolution:
                self.env.draw(delay)
            self.env.execute(action)
            reinforcement = self.env.reinforcement()
            state = self.env.getState()
            self.actor.createSAPs(state, self.env.generateActions())
            validActions = self.env.generateActions()
            action = self.actor.findNextAction(state, validActions, 0)
        if visualizeSolution:
            self.env.draw()


if __name__ == '__main__':
    type = 0
    size = 5
    initial = [(2,1)] # start with hole in (r,c)
    random = 0 # remove random pegs
    env = Board(type, size, initial, random)
    env.draw()
    visualizeSolution = False
    delay = 0.5

    alpha = 0.01
    lam = 0.9  #lambda
    gamma = 0.9
    eps = 1
    epsDecay = 0.99
    criticValuation = 1 # neural net valuation of states.
    nodesInLayers = [5,5,5]
    agent = Agent(env, alpha, alpha, lam, eps, gamma, criticValuation, nodesInLayers)

    agent.learn(500)
    agent.critic.display_useful_stuff()
    #agent.runGreedy(visualizeSolution, delay)
