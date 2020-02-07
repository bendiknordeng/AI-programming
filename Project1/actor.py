from env import Board
import random

#policy updates occur within the actor
#The actor should map from state, action to z (a real number)
#The actor must keep track of the results of performing actions in states.
class Actor:
    def __init__(self, alphaActor, eps, gamma):
        self.alphaActor = alphaActor
        self.eps = eps
        self.gamma = gamma
        self.saps = {}
        self.eligs = {}
        self.nextAction = None

    def createSAPs(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.saps.get((state,(fromP,toP))) == None:
                    self.saps[(state,(fromP,toP))] = 0

    def findNextAction(self, nextState):
        currentBest = -100000000000000
        switched = False
        for state, action in self.saps:

            if state == nextState:
                #print(state, action, self.saps[state,action])
                if self.saps[(state, action)] > currentBest:
                    self.nextAction = action
                    currentBest = self.saps[(state, action)]
                    switched = True
                elif self.saps[(state, action)] == currentBest:
                    if random.random() >= 0.5:
                        self.nextAction = action
                        currentBest = self.saps[(state, action)]
                        switched = True
        return self.nextAction

    def getAction(self):
        return self.nextAction

    def updateSAPs(self, surprise):
        alpha = self.alphaActor
        for stateAction in self.saps:
            #print(stateAction,self.saps[stateAction])
            e_s = self.eligs[stateAction]
            self.saps[stateAction] = self.saps[stateAction] + alpha*surprise*e_s

    def createEligibilities(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.eligs.get((state,(fromP,toP))) == None:
                    self.eligs[(state,(fromP,toP))] = 0

    def updateNextEligibility(self, state, action):
        self.eligs[(state, action)] = 1

    def updateEligibilities(self):
        gamma =self.gamma
        eps = self.eps
        for stateAction in self.eligs:
                self.eligs[stateAction] = gamma*eps*self.eligs[stateAction]
