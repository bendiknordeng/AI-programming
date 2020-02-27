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

    def get_best_child(self, leaf_search=False, verbose=False):
        assert self.children, "Current node does not have children"
        values = {}
        for child in self.children:
            if leaf_search:
                values[child] = (
                    child.Q + child.u) if self.turn else (child.Q - child.u)
            else:
                values[child] = child.Q
        chosen = max(values, key=values.get) if self.turn else min(
            values, key=values.get)
        if verbose:
            print("P{} is choosing".format(1 if self.turn else 2))
            for n in values:
                print("child: {}, Q: {:.2f}, u: {:.2f}, total value: {:.2f}, visits: {}, E: {}".format(
                    n.num_child, n.Q, n.u, n.Q + n.u if n.parent.turn else n.Q - n.u, n.visits, n.E))
            print("Chosen: {}\n".format(chosen))
        return chosen

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
