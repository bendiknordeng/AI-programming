
class Critic:
    def __init__(self, alphaCritic, eps, gamma):
        self.alphaCritic = alphaCritic
        self.eps = eps
        self.gamma = gamma
        self.surprise = 0
        self.values = {}

    def createStateValues(self, state):
        if self.values.get(state) == None:
            self.values[state] = 0

    def assignTDError(self, reinforcement, lastState, nextState):
        #print("inside TDError")
        #print(reinforcement)
        #print(self.gamma)
        #print(self.values[nextState])
        #print(self.values[lastState])
        self.surprise = reinforcement + self.gamma*self.values[nextState] - self.values[lastState]
        #print("TDError",self.TDError)
        #print()

    def getTDError(self):
        return self.surprise

    def updateStateValue(self, lastState):
        alpha = self.alphaCritic
        surprise = self.surprise
        self.values[lastState] = self.values[lastState] + alpha*surprise


    def updateLastEligibility(self, lastState):
        pass

if __name__ == '__main__':
    pass
