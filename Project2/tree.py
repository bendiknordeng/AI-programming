import math


class Node:
    def __init__(self, turn, state, num_child=0, parent=None, prev_action=None):
        self.turn = turn
        self.state = state
        self.children = []
        self.visits = 0
        self.parent = parent
        self.num_child = num_child
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
        parents = 1
        current = self
        while current.parent:
            parents += 1
            current = current.parent
        return parents


    def get_best_uct(self):
        assert self.children, "Current node does not have children"
        values = {}
        for child in self.children:
            values[child] = (child.Q + child.u) if self.turn else (child.Q - child.u)
        return max(values, key=values.get) if self.turn else min(
            values, key=values.get)

    def get_best_child(self):
        assert self.children, "Current node does not have children"
        values = {}
        for child in self.children:
            values[child] = child.visits
        return max(values, key=values.get)

    def get_node_id(self): # Returns which path is taken from root to given node
        current = self
        id = "_"+str(current.num_child)
        while True:
            current = current.parent
            id = "_"+str(current.num_child) + id
            if current.parent == None:
                return id

    def __str__(self):
        turn = 1 if self.turn else 2
        node = self.count_parents()
        s = "P{} Node: {} ({}) Child: {}, Visits: {} value: {:.3f}, E: {}"
        return s.format(turn, self.get_node_id(),node, self.num_child, self.visits, self.Q + self.u if self.parent.turn else self.Q - self.u, self.E)

    def __repr__(self):
        return "P{}_child{}_level{}".format(1 if self.turn else 2, self.num_child, self.count_parents())
