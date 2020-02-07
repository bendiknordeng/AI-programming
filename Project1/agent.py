from actor import Actor
from critic import Critic
from env import Board
import numpy as np
import time
from progressbar import ProgressBar
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, type, size, removePegs, alphaActor, alphaCritic, lambdod, gamma, eps, epsDecay):
        self.boardType = type
        self.boardSize = size
        self.removePegs = removePegs
        self.board = None
        self.alphaActor = alphaActor
        self.alphaCritic = alphaCritic
        self.eps = eps
        self.epsDecay = epsDecay
        self.lambdod = lambdod
        self.gamma = gamma
        self.actor = Actor(self.alphaActor, self.lambdod, self.gamma)
        self.critic = Critic(self.alphaCritic, self.lambdod, self.gamma)

    def resetBoard(self):
        type = self.boardType
        size = self.boardSize
        removePegs = self.removePegs
        self.board = Board(type,size)
        self.board.removePegs(removePegs)

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
            eps = eps*epsDecay

            self.resetBoard()
            reinforcement = 0
            #initialize new state values and (s,a)-pairs
            state = self.board.stringBoard()
            self.critic.createEligibility(state)
            self.critic.createStateValues(state)
            validActions = self.board.generateActions()
            self.actor.createSAPs(state, validActions)
            self.actor.createEligibilities(state, validActions)
            action = self.actor.findNextAction(state, eps)
            while len(validActions) > 0 :
                #make move
                self.board.jumpPegFromTo(action[0],action[1])
                lastAction = action
                lastState = state
                state = self.board.stringBoard()
                validActions = self.board.generateActions()
                #initialize new state values and (s,a)-pairs underway
                self.critic.createEligibility(state)
                self.critic.createStateValues(state)
                self.actor.createSAPs(state, validActions)
                self.actor.createEligibilities(state, validActions)
                reinforcement = self.board.reinforcement()
                action = self.actor.findNextAction(state, eps)
                if not action == -1:
                    self.actor.updateNextEligibility(state, action)
                surprise = self.critic.findTDError(reinforcement, lastState, state)
                self.critic.updateCurrentEligibility(lastState)
                self.critic.updateStateValues()
                self.critic.updateEligibilities()
                self.actor.updateSAPs(surprise)
                self.actor.updateEligibilities()
            pegsLeft.append(self.board.numberOfPegsLeft())
            iterationNumber.append(i)
            #epsList.append(eps)

        timeSpend = time.time()- start_time
        print("time spend", timeSpend)
        plt.plot(iterationNumber, pegsLeft);
        plt.show()
        #plt.plot(iterationNumber,epsList)
        #plt.show()

    def runGreedy(self):
        start_time = time.time()
        self.resetBoard()
        #self.board.draw()
        reinforcement = 0
        state = self.board.stringBoard()
        action = self.actor.findNextAction(state,0)
        while len(self.board.generateActions()) > 0:
            #self.board.draw(0.5)
            self.board.jumpPegFromTo(action[0],action[1])
            reinforcement = self.board.reinforcement()
            state = self.board.stringBoard()
            self.actor.createSAPs(state, self.board.generateActions())
            action = self.actor.findNextAction(state,0)
        self.board.draw()

def main():
    alpha = 0.85
    lambdod = 0.9  #lambda
    gamma = 0.95
    eps = 1
    epsDecay = 0.99
    type = 0
    size = 5
    removePegs = [(2,0)]
    runs = 300
    agent = Agent(type, size, removePegs, alpha, alpha, lambdod, gamma, eps, epsDecay)
    agent.learn(runs)
    agent.runGreedy()

main()
