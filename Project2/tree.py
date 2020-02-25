import math


class Node:
    def __init__(self, turn, state, parent=None, prev_action=None):
        self.turn = turn
        self.state = state
        self.children = []
        self.visits = 0
        self.parent = parent
        self.actions = []
        self.prev_action = prev_action
        self.E = 0  # evaluation with respect to wins
        self.Q = 0
        self.u = 0
        
    def add_child(self, action, child):
        self.actions.append(action)
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
            values[child] = (
                child.Q + child.u) if self.turn else (child.Q - child.u)
        return max(values, key=values.get) if self.turn else min(values, key=values.get)

    def __str__(self):
        return "Turn: Player {}".format(1 if self.turn else 2) + "\nState: {}".format(self.state) + "\nVisits: {}\n".format(self.visits)

    def __repr__(self):
        return "Player_{}-Node_{}-Visits_{}".format(1 if self.turn else 2, self.count_parents(), self.visits)

# class Edge:
#    def __init__(self, action, parent, child):
#        self.parent = parent
#        self.child = child
#        self.action = action
#        self.visits = 0
#        self.value = 0
#        child.set_prev_action(self)
#        parent.add_child(self, child)
#
#    def update_value(self, reinforcement):
#        self.value += reinforcement
#
#    def __repr__(self):
#        return str(self.action)
