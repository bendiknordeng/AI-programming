import random
import numpy as np
from collections import defaultdict


class Tree:
    def __init__(self):
        self.stateToNode = {} #key: (player, boardState), value: Node-object

    def getNode(self, state): # lookup Node in tree
        if self.hasState(state):
            return self.stateToNode[state] #state is list [player, boardState]
        return None

    def hasState(self, state):
        return self.stateToNode.get(state) != None

    def addState(self, state, legal_actions):
        if not self.hasState(state):
            self.stateToNode[state] = Node(state, legal_actions)

    def defaultPolicy(self, board):
        return random.choice(board.get_legal_actions()) #choose random action

    def treePolicy(self, state, c):
        import pdb; pdb.set_trace()
        node = self.getNode(state)
        actions = list(node.actions.keys())
        bestAction = random.choice(actions)
        bestValue = -1 * np.infty if state[0] == 1 else np.infty
        for action in actions:
            actionValue = node.getActionValue(action, c)
            if state[0] == 1: # player 1
                if actionValue >= bestValue:
                    bestValue = actionValue
                    bestAction = action
            else: #player 2
                if actionValue <= bestValue:
                    bestValue = actionValue
                    bestAction = action
        return bestAction

    def backup(self, nodes, z):
        for node in nodes:
            node.incrementVisit()
            node.incrementLastAction()
            node.updateQ(z)


class Node:
    def __init__(self, state, legal_actions):
        self.player, self.state = state
        self.nVisits = 1
        self.actions = {} # key: action, value: [number_of_times_chosen, q_value]
        for action in legal_actions:
            self.actions[action] = [0,0]
        self.prev_action = None

    def incrementVisit(self):
        self.nVisits += 1

    def incrementLastAction(self):
        self.actions[self.prev_action][0] += 1

    def setLastAction(self, action):
        self.prev_action = action


    def updateQ(self, z):
        self.actions[self.prev_action][1] += (z - self.actions[self.prev_action][1])/(self.actions[self.prev_action][0])

    def getActionValue(self, action, c):
        nVisits = self.nVisits
        nChosen, q = self.actions[action]
        if self.player == 1: #player 1
            return q + c * np.sqrt(np.log(nVisits)/(nChosen+1)) #+1 in case of nChosen == 0
        else: #player 2
            return q - c * np.sqrt(np.log(nVisits)/(nChosen+1)) #+1 in case of nChosen == 0

    def __repr__(self):
        return str({"Node",self.state, self.player})
