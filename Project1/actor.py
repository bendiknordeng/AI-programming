import random
import numpy as np

# policy updates occur within the actor
# The actor should map from state, action to z (a real number)
# The actor must keep track of the results of performing actions in states.

class Actor:
    def __init__(self, alphaActor, lam, gamma):
        self.alphaActor = alphaActor
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
        alpha = self.alphaActor
        for sap in self.saps:
            e_s = self.eligs[sap]
            if e_s > 0:
                self.saps[sap] = self.saps[sap] + alpha * td_error * e_s

    def createEligibilities(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.eligs.get((state, (fromP, toP))) == None:
                    self.eligs[(state, (fromP, toP))] = 0

    def updateCurrentEligibility(self, state, action):
        self.eligs[(state, action)] = 1

    def updateEligibilities(self):
        gamma = self.gamma
        lam = self.lam
        for stateAction in self.eligs:
            self.eligs[stateAction] = gamma * lam * self.eligs[stateAction]

    def resetEligibilities(self):
        for state in self.eligs:
            self.eligs[state] = 0
