import math


class Node:
    def __init__(self, turn, state, parent=None, prev_action=None):
        self.turn = turn
        self.state = state
        self.children = []
        self.visits = 0
        self.parent = parent
        self.prev_action = prev_action
        self.E = 0  # evaluation with respect to wins
        self.Q = 0
        self.u = 0

    def add_child(self, child):
        self.children.append(child)

    def update_values(self, reinforcement, c):
        self.E += reinforcement
        self.Q = self.E / self.visits
        self.u = c * \
            math.sqrt(math.log(self.parent.visits) / (1 + self.visits))

    def count_parents(self):
        parents = 0
        current = self
        while current.parent:
            parents += 1
            current = current.parent
        return parents

    def get_best_child(self):
        values = {}
        for child in self.children:
            values[child] = (child.Q + child.u) if self.turn else (child.Q - child.u)
        return max(values, key=values.get) if self.turn else min(values, key=values.get)

    def __str__(self):
        turn = 1 if self.turn else 2
        node = self.count_parents()
        s = "P{} Node: {:>2} Visits: {:>5} Q: {:>7.1f} u: {:.3f} E: {:>6}"
        return s.format(turn, node, self.visits, self.Q, self.u, self.E)
