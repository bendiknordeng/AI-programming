import random
import numpy as np
from collections import defaultdict

class Node:
    def __init__(self, state, parent=None, prev_action=None):
        self.state = state
        self.children = []
        self._number_of_visits = 0 # visits
        self._results = defaultdict(int)
        self.parent = parent
        self.prev_action = prev_action
        self._untried_actions = None

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((np.log(self.n) /(1+c.n)))
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
        wins = self._results[self.parent.state.turn]
        losses = self._results[-1 * self.parent.state.turn]
        return (wins - losses) / self.n

    @property
    def n(self):
        return self._number_of_visits

    @property
    def game_state(self):
        return self.state.state

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = Node(next_state, parent=self, prev_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
