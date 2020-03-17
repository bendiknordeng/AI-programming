import random
import numpy as np
from collections import defaultdict

class Tree:
    def __init__(self):
        self.state_to_node = {} #key: (player, boardState), value: Node-object

    def get_node(self, env): # lookup Node in tree
        state = env.get_state()
        if self.state_to_node.get(state):
            return self.state_to_node[state] #state is list [player, boardState]
        else:
            self.state_to_node[state] = Node(state[0], env.get_legal_actions())
            return False

    def rollout_policy(self, board):
        return random.choice(board.get_legal_actions()) #choose random action

    def tree_policy(self, env, c):
        player, _ = env.get_state()
        node = self.get_node(env)
        actions = list(node.actions.keys())
        best_action = random.choice(actions)
        best_value = -np.infty if player == 1 else np.infty
        for action in actions:
            action_value = node.get_action_value(action, c)
            if player == 1:
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            else:
                if action_value < best_value:
                    best_value = action_value
                    best_action = action
        return best_action

    def backup(self, nodes, z):
        for node in nodes:
            node.update_values(z)

class Node:
    def __init__(self, player, legal_actions):
        self.player = player
        self.visits = 1
        self.actions = {} # key: action, value: visits, results
        for action in legal_actions:
            self.actions[action] = [0, 0]
        self.prev_action = None

    def update_values(self, z):
        self.visits += 1
        self.actions[self.prev_action][0] += 1
        self.actions[self.prev_action][1] += z

    def set_last_action(self, action):
        self.prev_action = action

    def q(self, action):
        n, z = self.actions[action]
        if n == 0:
            return 0
        else:
            return z/n

    def get_action_value(self, action, c):
        n = self.actions[action][0]
        q = self.q(action)
        if self.player == 1: #player 1
            return q + c * np.sqrt(np.log(self.visits)/(n+1)) #+1 in case n == 0
        else: #player 2
            return q - c * np.sqrt(np.log(self.visits)/(n+1)) #+1 in case n == 0

    def __repr__(self):
        return "Player: {}, Visits: {}, Actions: {}, Previous action: {}".format(self.player, self.visits, self.actions, self.prev_action)
