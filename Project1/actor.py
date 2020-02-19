import random
import numpy as np

class Actor:
    def __init__(self, alpha, lam, gamma):
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.saps = {}
        self.eligs = {}

    # create state action pair for given state and actions
    def createSAPs(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.saps.get((state, (fromP, toP))) == None: # if given SAP is not yet created
                    self.saps[(state, (fromP, toP))] = 0

    # assign value of state action pairs, and choose greedy with prob 1-eps, else choose random
    def findNextAction(self, state, actions, eps):
        actionStack = {}
        for fromP in actions:
            for toP in actions[fromP]:
                actionStack[(fromP, toP)] = self.saps[(state, (fromP, toP))]
        if len(actionStack) > 0:
            if random.random() < eps:
                return random.choice(list(actionStack.items()))[0]
            else:
                return max(actionStack, key=actionStack.get)

    # based on td_error, update state action pairs with learning rate alpha
    def updateSAPs(self, td_error):
        for sap in self.eligs:
            if sap[1] != None: # last state has no valid actions
                self.saps[sap] = self.saps[sap] + self.alpha * td_error * self.eligs[sap]

    # assign 1 to last chosen state action pair
    def updateEligibility(self, state, action):
        self.eligs[(state, action)] = 1

    # decay eligibilities with factor gamma*lambda
    def decayEligibilities(self):
        for sap in self.eligs:
            self.eligs[sap] = self.gamma * self.lam * self.eligs[sap]

    # remove all eligibilities to make new trace
    def resetEligibilities(self):
        self.eligs.clear()
