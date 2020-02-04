import random

class Critic:
    def __init__(self, agent, alphaCritic, eps, gamma):
        self.agent = agent
        self.alphaCritic = alphaCritic
        self.eps = eps
        self.gamma = gamma
        self.lastState = self.agent.getState()
        self.nextState = None
        self.values = {}
        self.values[self.lastState] = random.random() #initializing with initial state

    def getNextState(self):
        self.nextState = self.agent.getState()

    def createNewValues(self):
        if self.values.get(self.nextState) == None:
            self.values[self.nextState] = random.random()

    def updateValues(self):
        self.nextState = self.agent.getState()

    def TDerror(self):
        return self.agent.getReward() + self.gamma*self.values[self.nextState] - self.values[self.lastState]

if __name__ == '__main__':
    pass
