import random
import numpy as np

class Tree:
    def __init__(self):
        self.game_mode = 0
        self.stateToNode = {} #key: (player, boardState), value: Node-object

    def setGameMode(self, game_mode):
        self.game_mode = game_mode

    def getNode(self, state): # lookup Node in tree
        state = (state[0] , tuple(state[1])) if self.game_mode else state #make tuple if Ledge
        if self.hasState(state):
            return self.stateToNode[state] #state is list [player, boardState]
        return None

    def hasState(self, state):
        state = (state[0] , tuple(state[1])) if self.game_mode else state #make tuple if Ledge
        return self.stateToNode.get(state) != None

    def addState(self, state, legal_actions):
        state = (state[0] , tuple(state[1])) if self.game_mode else state #make tuple if Ledge
        if not self.hasState(state):
            self.stateToNode[state] = Node(state, legal_actions)

    def defaultPolicy(self, board):
        return random.choice(board.get_legal_actions()) #choose random action

    def treePolicy(self, state, c):
        state = (state[0] , tuple(state[1])) if self.game_mode else state #make tuple if Ledge
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
        nChosen, q = self.actions[self.prev_action]
        self.actions[self.prev_action][1] += (z - q)/(nChosen)

    def getActionValue(self, action, c):
        nVisits = self.nVisits
        nChosen, q = self.actions[action]
        if self.player == 1: #player 1
            return q + c * np.sqrt(np.log(nVisits)/(nChosen+1)) #+1 in case nChosen == 0
        else: #player 2
            return q - c * np.sqrt(np.log(nVisits)/(nChosen+1)) #+1 in case nChosen == 0
