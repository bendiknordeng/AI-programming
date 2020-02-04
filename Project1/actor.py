from env import Board

#policy updates occur within the actor
#The actor should map from state, action to z (a real number)
#The actor must keep track of the results of performing actions in states.
class Actor:
    def __init__(self, agent, alphaActor, eps, gamma):
        self.agent = agent
        self.alphaActor = alphaActor
        self.eps = eps
        self.gamma = gamma
        self.saps = {}
        self.createNewSAPs()

    def createNewSAPs(self):
        newState = self.agent.getState()
        possActions = self.agent.getActions()
        print(newState)
        for fromPos in possActions:
            for toPos in possActions[fromPos]:
                if self.saps.get((newState,(fromPos,toPos))) == None:
                    self.saps[newState,(fromPos,toPos)] = 0

    def getNextMove(self):
        currentBest = -100
        nextMove = ((-1,-1),(-1,-1))
        for state, action in self.saps:
            if self.saps[(state,action)] > currentBest:
                nextMove = action
        return nextMove

    def score(self, currentState, nextState):
        pass

if __name__ == '__main__':
    pass
