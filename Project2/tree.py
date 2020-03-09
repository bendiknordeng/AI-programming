import random
import numpy as np
from collections import defaultdict

class Node:
    wins = defaultdict(int)
    visits = defaultdict(int)

    def __init__(self, state, parent=None, prev_action=None):
        self.state = state
        self.children = []
        self.parent = parent
        self.prev_action = prev_action
        self._untried_actions = None

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            c.q + c_param * np.sqrt((np.log(self.n) /(1+c.n)))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        wins = self.wins[(self.parent.game_state, self.prev_action)]
        return wins / self.n

    @property
    def n(self):
        return self.visits[self.game_state]

    @property
    def game_state(self):
        return self.state.state

    @property
    def player(self):
        return self.state.turn

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = Node(next_state, parent=self, prev_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        state = self.state
        while not state.is_game_over():
            possible_moves = state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            state = state.move(action)
        return state.game_result

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def backpropagate(self, result):
        self.visits[self.game_state] += 1
        if self.parent:
            self.wins[(self.parent.game_state, self.prev_action)] += 1 if self.parent.player == result else -1
            self.parent.backpropagate(result)

    def __repr__(self):
        return str({"Action": self.prev_action, "Q": self.q, "n": self.n})
