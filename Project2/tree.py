class Node:
    def __init__(self, turn, state, parent = None):
        self.turn = turn
        self.state = state
        self.children = []
        self.visits = 0
        self.parent = parent
        self.actions = []

    def set_prev_action(self, edge):
        self.prev_action = edge

    def add_child(self, action, child):
        self.actions.append(action)
        self.children.append(child)

    def __str__(self):
        return "Turn: Player {}".format(1 if self.turn else 2) + "\nState: {}".format(self.state) + "\nVisits: {}".format(self.visits)

    def __repr__(self):
        return str(self.state)

class Edge:
    def __init__(self, action, parent, child):
        self.parent = parent
        self.child = child
        self.action = action
        self.visits = 0
        self.value = 0

    def update_value(self, reinforcement):
        self.value += reinforcement

    def __repr__(self):
        return str(self.action)
