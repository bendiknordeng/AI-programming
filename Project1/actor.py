import random
import numpy as np

# policy updates occur within the actor
# The actor should map from state, action to z (a real number)
# The actor must keep track of the results of performing actions in states.

class Actor:
    def __init__(self, alphaActor, lambdod, gamma):
        self.alphaActor = alphaActor
        self.lambdod = lambdod
        self.gamma = gamma
        self.saps = {}
        self.eligs = {}

    def createSAPs(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.saps.get((state, (fromP, toP))) == None:
                    self.saps[(state, (fromP, toP))] = 0

    def findNextAction(self, currentState, validActions, eps):
        currentBest = np.NINF
        actionStack = {}
        for fromP in validActions:
            for toP in validActions[fromP]:
                actionStack[(fromP, toP)] = self.saps[(currentState, (fromP, toP))]

        if len(actionStack) > 0:
            if random.random() < eps:
                return random.choice(list(actionStack.items()))[0]
            else:
                return max(actionStack, key=actionStack.get)

    def updateSAPs(self, surprise):
        alpha = self.alphaActor
        for stateAction in self.saps:
            # print(stateAction,self.saps[stateAction])
            e_s = self.eligs[stateAction]
            self.saps[stateAction] = self.saps[stateAction] + \
                alpha * surprise * e_s

    def createEligibilities(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.eligs.get((state, (fromP, toP))) == None:
                    self.eligs[(state, (fromP, toP))] = 0

    def updateNextEligibility(self, state, action):
        self.eligs[(state, action)] = 1

    def updateEligibilities(self):
        gamma = self.gamma
        lambdod = self.lambdod
        for stateAction in self.eligs:
            self.eligs[stateAction] = gamma * lambdod * self.eligs[stateAction]
