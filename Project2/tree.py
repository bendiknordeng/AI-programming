import random
import numpy as np


class Tree:
    def __init__(self):
        self.state_to_node = {}  # key: (player, state), value: Node-object

    def get_node(self, env):  # lookup Node in tree
        state = env.get_state()
        if self.state_to_node.get(state):
            return self.state_to_node[state]
        else:
            self.state_to_node[state] = Node(state[0], env.get_legal_actions())
            return False

    def rollout_policy(self, env):
        return random.choice(env.get_legal_actions())  # choose random action

    def tree_policy(self, env, c):
        player, _ = env.get_state()
        node = self.get_node(env)
        actions = list(node.actions.keys())
        best_action = random.choice(actions)
        best_value = -np.infty if player == 1 else np.infty
        action_values = [node.get_action_value(a, c) for a in actions]
        return actions[np.argmax(action_values) if player == 1 else np.argmin(action_values)]

    def backup(self, nodes, result):
        for node in nodes:
            node.update_values(result)


class Node:
    def __init__(self, player, legal_actions):
        self.player = player
        self.visits = 1
        self.actions = {}  # key: action, value: [visits, q_value]
        for action in legal_actions:
            self.actions[action] = [0, 0]
        self.prev_action = None

    def update_values(self, result):
        self.visits += 1
        self.actions[self.prev_action][0] += 1
        n, q = self.actions[self.prev_action]
        self.actions[self.prev_action][1] += (result - q) / n

    def set_last_action(self, action):
        self.prev_action = action

    def get_action_value(self, action, c):
        n, q = self.actions[action]
        if self.player == 1:
            return q + c * np.sqrt(np.log(self.visits) / (n + 1))
        else:
            return q - c * np.sqrt(np.log(self.visits) / (n + 1))

    def __repr__(self):
        return "Player: {}, Visits: {}, Actions: {}, Previous action: {}".format(self.player, self.visits, self.actions, self.prev_action)
