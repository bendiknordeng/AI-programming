from actor import Actor
from critic import Critic
from env import Board
import numpy as np
import time
from progressbar import ProgressBar

class Agent:
    def __init__(self, type, size, removePegs, alphaActor, alphaCritic, eps, gamma):
        self.boardType = type
        self.boardSize = size
        self.removePegs = removePegs
        self.board = None
        self.alphaActor = alphaActor
        self.alphaCritic = alphaCritic
        self.eps = eps
        self.gamma = gamma
        self.actor = Actor(self.alphaActor, self.eps, self.gamma)
        self.critic = Critic(self.alphaCritic, self.eps, self.gamma)

    def resetBoard(self):
        type = self.boardType
        size = self.boardSize
        removePegs = self.removePegs
        self.board = Board(type,size)
        self.board.removePegs(removePegs)

    def learn(self, runs):
        endResults = []
        start_time = time.time()
        pbar = ProgressBar()
        for i in pbar(range(runs)):
            self.resetBoard()
            reinforcement = 0
            #initialize new state values and (s,a)-pairs for start (s,a)
            state = self.board.stringBoard()
            self.critic.createEligibility(state)
            self.critic.createStateValues(state)
            validActions = self.board.generateActions()
            self.actor.createSAPs(state, validActions)
            self.actor.createEligibilities(state, validActions)
            action = self.actor.findNextAction(state)
            #action = self.actor.getAction()
            while reinforcement == 0:
                #make move
                #self.board.draw()
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
                #do variable assignments after move
                reinforcement = self.board.reinforcement()
                action = self.actor.findNextAction(state)
                self.actor.updateNextEligibility(state, action)
                surprise = self.critic.findTDError(reinforcement, lastState, state)

                self.critic.updateCurrentEligibility(lastState)

                self.critic.updateStateValues()
                self.critic.updateEligibilities()
                self.actor.updateSAPs(surprise)
                self.actor.updateEligibilities()
            endResults.append(reinforcement)
        print("time spend", time.time()- start_time)
        return endResults

    def runGreedy(self):
        start_time = time.time()

        self.resetBoard()
        self.board.draw()
        reinforcement = 0
        state = self.board.stringBoard()
        action = self.actor.findNextAction(state)
        while reinforcement == 0:
            print("chose action", action)
            self.board.draw(0.5)
            self.board.jumpPegFromTo(action[0],action[1])

            reinforcement = self.board.reinforcement()
            state = self.board.stringBoard()
            self.actor.createSAPs(state, self.board.generateActions())
            action = self.actor.findNextAction(state)


        self.board.draw()

def main():
    alpha = 0.85
    eps = 0.9  #lambda
    gamma = 0.95
    type = 0
    size = 6
    removePegs = [(2,0)]
    runs = 1
    agent = Agent(type, size, removePegs, alpha, alpha, eps, gamma)
    agent.learn(10000)
    agent.runGreedy()

main()
