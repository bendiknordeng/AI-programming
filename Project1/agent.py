from actor import Actor
from critic import Critic
from env import Board
import numpy as np
import time
from progressbar import ProgressBar
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, env, alphaActor, alphaCritic, eps, gamma):
        self.env = env
        self.alphaActor = alphaActor
        self.alphaCritic = alphaCritic
        self.eps = eps
        self.epsDecay = epsDecay
        self.lam = lam
        self.gamma = gamma
        self.actor = Actor(self.alphaActor, self.lam, self.gamma)
        self.critic = Critic(self.alphaCritic, self.lam, self.gamma)

    def learn(self, runs):
        eps = self.eps
        epsDecay = self.epsDecay
        pegsLeft = []
        iterationNumber = []
        #epsList=[]
        iteration = 0
        start_time = time.time()
        pbar = ProgressBar()
        for i in pbar(range(runs)):
            iteration += 1
            eps = eps * epsDecay
            self.env.reset()
            self.actor.resetEligibilities()
            self.critic.resetEligibilities()
            # initialize new state values and (s,a)-pairs for start (s,a)
            state = self.env.getState()
            self.critic.createEligibility(state)
            self.critic.createStateValues(state)

            validActions = self.env.generateActions()
            self.actor.createSAPs(state, validActions)
            self.actor.createEligibilities(state, validActions)
            action = self.actor.findNextAction(state, validActions, eps)
            while len(validActions) > 0:
                lastState = state # save current state before new action
                self.env.execute(action)
                state = self.env.getState()
                validActions = self.env.generateActions()

                self.critic.createEligibility(state)
                self.critic.createStateValues(state)
                self.actor.createSAPs(state, validActions)
                self.actor.createEligibilities(state, validActions)

                reinforcement = self.env.reinforcement()
                action = self.actor.findNextAction(state, validActions, eps)

                self.actor.updateCurrentEligibility(state, action)
                td_error = self.critic.findTDError(reinforcement, lastState, state)

                self.critic.updateCurrentEligibility(lastState)
                self.critic.updateStateValues()
                self.critic.updateEligibilities()
                self.actor.updateSAPs(td_error)
                self.actor.updateEligibilities()

            pegsLeft.append(self.env.numberOfPegsLeft())
            iterationNumber.append(i)
            #epsList.append(eps)
        print(len(self.actor.saps))

        time_spent = time.time() - start_time
        print("Time spent", time_spent)
        plt.plot(iterationNumber, pegsLeft)
        plt.show()
        #plt.plot(iterationNumber,epsList)
        #plt.show()

    def runGreedy(self):
        start_time = time.time()
        self.env.reset()
        self.env.draw()
        reinforcement = 0
        state = self.env.getState()
        validActions = self.env.generateActions()
        action = self.actor.findNextAction(state, validActions, 0)
        while len(validActions) > 0:
            #self.env.draw(0.5)
            self.env.execute(action)
            reinforcement = self.env.reinforcement()
            state = self.env.getState()
            self.actor.createSAPs(state, self.env.generateActions())
            validActions = self.env.generateActions()
            action = self.actor.findNextAction(state, validActions, 0)
        self.env.draw()


if __name__ == '__main__':
    alpha = 0.85 # learning rate
    lam = 0.9  # trace decay
    gamma = 0.95 # discount factor
    eps = 1 # epsilon
    epsDecay = 0
    type = 1 # type 0 = triangle, type 1 = diamond
    size = 4
    initial = [(2,2)]  # start with hole in (r,c)
    random = 0 # remove random pegs
    runs = 100

    env = Board(type, size, initial, random)
    agent = Agent(env, alpha, alpha, eps, gamma)
    agent.learn(runs)
    agent.runGreedy()
