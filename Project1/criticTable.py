import random

class CriticTable:
    def __init__(self, alpha, lam, gamma):
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.values = {}
        self.eligs = {}  # key, value is state, eligibility

    def createStateValues(self, state):
        if self.values.get(state) == None:
            self.values[state] = random.random()

    def findTDError(self, reinforcement, lastState, state):
        self.td_error = reinforcement + self.gamma * self.values[state] - self.values[lastState]
        return self.td_error

    def updateStateValues(self):
        for state in self.eligs:
            self.values[state] = self.values[state] + self.alpha * self.td_error * self.eligs[state]

    def createEligibility(self, state):
        if self.eligs.get(state) == None:
            self.eligs[state] = 1

    def updateEligibilities(self):
        for state in self.eligs:
            self.eligs[state] = self.gamma * self.lam * self.eligs[state]

    def resetEligibilities(self):
        self.eligs.clear()

    def stateValue(self, state):
        return self.values[state]
