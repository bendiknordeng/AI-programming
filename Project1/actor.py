import random
import numpy as np

#policy updates occur within the actor
#The actor should map from state, action to z (a real number)
#The actor must keep track of the results of performing actions in states.
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
                if self.saps.get((state,(fromP,toP))) == None:
                    self.saps[(state,(fromP,toP))] = 0

    def findNextAction(self, nextState, eps):
        currentBest = np.NINF
        actionStack = []
        #print()
        #print("action, score")
        for state, action in self.saps:
            if state == nextState:
                #print(action,self.saps[(state,action)])
                if len(actionStack) > 0:
                    appended = False
                    for i in range(len(actionStack)):
                        a = actionStack[i]
                        if self.saps[(state, action)] <= self.saps[(state, a)]: #insert action in first position where it is \leq
                            actionStack.insert(i, action)
                            appended = True
                            break

                    if not appended:
                        actionStack.append(action)
                else:
                    actionStack.append(action)

        if len(actionStack) > 0:
            if random.random() < eps:
                i = np.random.randint(0, len(actionStack))
                #print("chose random action:", actionStack[i])
                return actionStack[i]
            else:
                action = actionStack.pop()
                #print("chose greedy action:", action )
                return action
        else:
            #print("chose dummy action")
            return -1 #return dummy move

    def updateSAPs(self, surprise):
        alpha = self.alphaActor
        for stateAction in self.saps:
            #print(stateAction,self.saps[stateAction])
            e_s = self.eligs[stateAction]
            self.saps[stateAction] = self.saps[stateAction] + alpha*surprise*e_s

    def createEligibilities(self, state, actions):
        for fromP in actions:
            for toP in actions[fromP]:
                if self.eligs.get((state,(fromP,toP))) == None:
                    self.eligs[(state,(fromP,toP))] = 0

    def updateNextEligibility(self, state, action):
        self.eligs[(state, action)] = 1

    def updateEligibilities(self):
        gamma =self.gamma
        lambdod = self.lambdod
        for stateAction in self.eligs:
                self.eligs[stateAction] = gamma*lambdod*self.eligs[stateAction]
