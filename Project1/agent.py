from actor import Actor
from critic import Critic
from env import Board
import numpy as np

class Agent:
    def __init__(self, type, size, removePegs, alphaActor, alphaCritic, eps, gamma):
        self.boardType = type
        self.boardSize = size
        self.removePegs = removePegs
        self.board = None
        self.resetBoard()
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

    def getActorsAction(self):
        return self.actor.getNextAction()

    def actorFindNextAction(self):
        self.actor.findNextAction(getActions())

    def learn(self):
        endResults = []
        for i in range(10000):
            self.resetBoard()
            reinforcement = 0
            if i == 9999:
                self.board.draw()

            #initialize for start (s,a)
            self.critic.createStateValues(self.getState())
            self.actor.createSAPs(self.getState(), self.getActions())
            self.actor.findNextAction(self.getState())
            while reinforcement == 0:
                action = self.actor.getAction()
                #save previousState
                lastState = self.getState()
                #make move
                if i == 9999:
                    self.board.draw(1)
                self.board.jumpPegFromTo(action[0],action[1])
                #initialize new state values and (s,a)-pairs underway
                self.critic.createStateValues(self.getState())
                self.actor.createSAPs(self.getState(), self.getActions())
                #do variable assignments after move
                reinforcement = self.getReinforcement()
                self.actor.findNextAction(self.getState())
                self.actor.updateEligibilities(self.getState())
                self.critic.assignTDError(reinforcement, lastState, self.getState())
                self.critic.updateLastEligibility(lastState)
                surprise = self.critic.getTDError()
                self.critic.updateStateValue(lastState)
                self.actor.updateSAP(lastState, action, surprise)
            endResults.append(reinforcement)
            #print()
            #print("state values")
            #for state in self.critic.values:
        #        print (state, self.critic.values[state])
            #print()
            #print("SAPs")
            #for state, action in self.actor.saps:
            #    print (state, action, self.actor.saps[state,action])
            #print()
            #print(endResults[-40:])
            if i == 9999:
                self.board.draw(1)
        return endResults

    def runGreedy(self):
        print("ready for greedy run")
        reinforcement = 0
        self.actor.findNextAction(self.getState())
        while reinforcement == 0:
            action = self.actor.getAction()
            #save previousState
            lastState = self.getState()
            #make move
            self.board.draw()
            self.board.jumpPegFromTo(action[0],action[1])
            #initialize new state values and (s,a)-pairs underway
            self.critic.createStateValues(self.getState())
            self.actor.createSAPs(self.getState(), self.getActions())
            #do variable assignments after move
            reinforcement = self.getReinforcement()
            self.actor.findNextAction(self.getState())
            self.actor.updateEligibilities(self.getState())
            self.critic.assignTDError(reinforcement, lastState, self.getState())
            self.critic.updateLastEligibility(lastState)
            surprise = self.critic.getTDError()
            self.critic.updateStateValue(lastState)
            self.actor.updateSAP(lastState, action, surprise)


    def getState(self):
        return self.board.stringBoard()

    def getActions(self):
        return self.board.generateValidMoves()

    def getReinforcement(self):
        return self.board.reinforcement()

if __name__ == '__main__':

    alpha = 0.85
    eps = 0.9 #lambda
    gamma = 0.95
    type = 0
    size = 6
    removePegs = [(1,0)]

    agent = Agent(type, size, removePegs, alpha, alpha, eps, gamma)
    print(agent.learn())
    agent.runGreedy()
