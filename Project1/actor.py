from env import Board
import random

#policy updates occur within the actor
#The actor should map from state, action to z (a real number)
#The actor must keep track of the results of performing actions in states.
class Actor:
    def __init__(self, alphaActor, eps, gamma):
        self.alphaActor = alphaActor
        self.eps = eps
        self.gamma = gamma
        self.saps = {}
        self.nextAction = None

    def createSAPs(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.saps.get((state,(fromP,toP))) == None:
                    self.saps[state,(fromP,toP)] = 0

    def findNextAction(self, nextState):
        currentBest = -1 * 10^6
        print("finding next action")
        for state, action in self.saps:
            if state == nextState:
                print(state, action,self.saps[(state, action)])
                if self.saps[(state, action)] > currentBest:
                    self.nextAction = action
                    currentBest = self.saps[(state, action)]
                elif self.saps[(state, action)] == currentBest:
                    if random.random() >= 0.5:
                        self.nextAction = action
                        currentBest = self.saps[(state, action)]
        print("chose action", self.nextAction)
        print()




    def getAction(self):
        return self.nextAction

    def updateSAP(self, lastState, action, surprise):
        alpha = self.alphaActor
        self.saps[(lastState, action)] = self.saps[(lastState, action)] + alpha*surprise

    def updateEligibilities(self, state):
        state = state
        action = self.nextAction
        pass

    def score(self, currentState, nextState):
        pass

if __name__ == '__main__':
    pass
