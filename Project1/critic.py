class CriticTable:
    def __init__(self, alphaCritic, lambdod, gamma, criticValuation = 0, inputDim = 0, nodesInLayers = [0]):
        self.alphaCritic = alphaCritic
        self.lam = lam
        self.gamma = gamma
        self.surprise = 0
        self.values = {}
        self.eligs = {}  # key, value is state, eligibility

    def createStateValues(self, state):
        if self.values.get(state) == None:
            self.values[state] = random.random()

    def findTDError(self, reinforcement, lastState, state):
        self.td_error = reinforcement + self.gamma * \
            self.values[state] - self.values[lastState]
        return self.td_error

    def getTDError(self):
        return self.td_error

    def updateStateValues(self):
        alpha = self.alphaCritic
        td_error = self.td_error
        for state in self.values:
            e_s = self.eligs[state]
            if e_s > 0:
                self.values[state] = self.values[state] + alpha * td_error * e_s

    def createEligibility(self, state):
        if self.stateValueEligibilities.get(state) == None:
            self.stateValueEligibilities[state] = 0

    def updateCurrentEligibility(self, lastState):
        self.stateValueEligibilities[lastState] = 1

    def updateEligibilities(self):
        gamma = self.gamma
        lam = self.lam
        for state in self.eligs:
            self.eligs[state] = gamma * lam * self.eligs[state]

    def resetEligibilities(self):
        for state in self.eligs:
            self.eligs[state] = 0
