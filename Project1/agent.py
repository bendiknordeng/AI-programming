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
        self.gamma = gamma
        self.actor = Actor(self.alphaActor, self.eps, self.gamma)
        self.critic = Critic(self.alphaCritic, self.eps, self.gamma)

    def learn(self, runs):
        pegsLeft = []
        iterationNumber = []
        iteration = 0
        start_time = time.time()
        pbar = ProgressBar()
        for i in pbar(range(runs)):
            iteration += 1
            self.env.reset()
            reinforcement = 0
            #initialize new state values and (s,a)-pairs for start (s,a)
            state = self.env.state()
            self.critic.createEligibility(state)
            self.critic.createStateValues(state)
            validActions = self.env.generateActions()
            self.actor.createSAPs(state, validActions)
            self.actor.createEligibilities(state, validActions)
            action = self.actor.findNextAction(state)
            #action = self.actor.getAction()
            while len(validActions) > 0 :
                #make move
                #self.env.draw()
                self.env.jumpPegFromTo(action[0],action[1])
                lastAction = action
                lastState = state
                state = self.env.state()
                validActions = self.env.generateActions()
                #initialize new state values and (s,a)-pairs underway
                self.critic.createEligibility(state)
                self.critic.createStateValues(state)
                self.actor.createSAPs(state, validActions)
                self.actor.createEligibilities(state, validActions)
                #do variable assignments after move
                reinforcement = self.env.reinforcement()
                action = self.actor.findNextAction(state)
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
        print("average time per iteration", timeSpend/runs)
        plt.plot(iterationNumber, pegsLeft);
        plt.show()

    def runGreedy(self):
        start_time = time.time()
        self.env.reset()
        self.env.draw()
        reinforcement = 0
        state = self.env.state()
        action = self.actor.findNextAction(state)
        while len(self.env.generateActions())>0:
            #print("chose action", action)
            self.env.draw(0.5)
            self.env.jumpPegFromTo(action[0],action[1])
            reinforcement = self.env.reinforcement()
            state = self.env.state()
            self.actor.createSAPs(state, self.env.generateActions())
            action = self.actor.findNextAction(state)
        self.env.draw()

if __name__ == '__main__':
    alpha = 0.85
    eps = 0.9  #lambda
    gamma = 0.95
    type = 0
    size = 4
    initial = [(2,0)] # start with hole in (r,c)
    random = 0 # remove random pegs

    env = Board(type, size, initial, random)
    agent = Agent(env, alpha, alpha, eps, gamma)
    agent.learn(400)
    agent.runGreedy()
