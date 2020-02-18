import random
import numpy as np

# policy updates occur within the actor
# The actor should map from state, action to z (a real number)
# The actor must keep track of the results of performing actions in states.

class Actor:
    def __init__(self, alpha, lam, gamma):
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.saps = {}
        self.eligs = {}

    def createSAPs(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.saps.get((state, (fromP, toP))) == None:
                    self.saps[(state, (fromP, toP))] = 0

    def findNextAction(self, currentState, validActions, eps):
        actionStack = {}
        for fromP in validActions:
            for toP in validActions[fromP]:
                actionStack[(fromP, toP)] = self.saps[(currentState, (fromP, toP))]

        if len(actionStack) > 0:
            if random.random() < eps:
                return random.choice(list(actionStack.items()))[0]
            else:
                return max(actionStack, key=actionStack.get)

    def updateSAPs(self, td_error):
        for sap in self.eligs:
            if sap[1] != None:
                self.saps[sap] = self.saps[sap] + self.alpha * td_error * self.eligs[sap]

    def updateEligibility(self, state, action):
        self.eligs[(state, action)] = 1

    def decayEligibilities(self):
        for sap in self.eligs:
            self.eligs[sap] = self.gamma * self.lam * self.eligs[sap]

    def resetEligibilities(self):
        self.eligs.clear()
