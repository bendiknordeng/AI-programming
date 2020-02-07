
class Critic:
    def __init__(self, alphaCritic, eps, gamma):
        self.alphaCritic = alphaCritic
        self.eps = eps
        self.gamma = gamma
        self.surprise = 0
        self.values = {}
        self.eligs = {} #key, value is state, eligibility

    def createStateValues(self, state):
        if self.values.get(state) == None:
            self.values[state] = 0

    def findTDError(self, reinforcement, lastState, state):
        self.surprise = reinforcement + self.gamma*self.values[state] - self.values[lastState]
        return self.surprise

    def getTDError(self):
        return self.surprise

    def updateStateValues(self):
        alpha = self.alphaCritic
        surprise = self.surprise
        for state in self.values:
            e_s =self.eligs[state]
            self.values[state] = self.values[state] + alpha*surprise*e_s

    def createEligibility(self, state):
        if self.eligs.get(state) == None:
            self.eligs[state] = 0

    def updateCurrentEligibility(self, lastState):
        self.eligs[lastState] = 1

    def updateEligibilities(self):
        gamma =self.gamma
        eps = self.eps
        for state in self.eligs:
            self.eligs[state] = gamma*eps*self.eligs[state]
