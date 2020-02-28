import math


class Node:
    def __init__(self, turn, state, parent=None, prev_action=None, num_child=0, is_root=False):
        self.turn = turn
        self.state = state
        self.children = []
        self.num_child = num_child
        self.visits = 0
        self.parent = parent
        self.prev_action = prev_action
        self.is_root = is_root
        self.E = 0
        self.Q = 0
        self.u = 0

    def add_child(self, child):
        self.children.append(child)

    def update_values(self, result, c):
        self.parent.visits += 1
        self.E += result
        self.Q = self.E / self.visits
        self.u = c * \
            math.sqrt(math.log(2*self.parent.visits) / self.visits)

    def reset(self):
        self.visits = 0
        self.children = []
        self.is_root = True
        self.E = 0
        self.Q = 0
        self.u = 0

    def count_parents(self):
        parents = 0
        current = self
        while current.parent:
            parents += 1
            current = current.parent
        return parents
        
    def __str__(self):
        turn = 1 if self.turn else 2
        s = "Player: {}\nRoot: {}\nLevel: {}\nChildren: {}\nVisits: {}\nWins: {}\n"
        return s.format(turn, self.is_root, self.count_parents(), self.children, self.visits, self.E)

    def __repr__(self):
        return "P{}_child{}_visits{}".format(1 if self.turn else 2, self.num_child, self.visits)
