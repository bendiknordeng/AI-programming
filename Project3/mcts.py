class MonteCarloTreeSearch:
    def __init__(self, node, ANN, eps):
        self.root = node
        self.ANN = ANN
        self.eps = eps

    def best_action(self, simulations):
        for i in range(simulations):
            leaf = self.tree_policy()
            result = leaf.rollout(self.ANN, self.eps)
            leaf.backpropagate(result)
        return self.root.best_child(c_param = 0)

    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
