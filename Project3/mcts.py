from tree import Tree

class MonteCarloTreeSearch:
    def __init__(self, ANN=None, c=1.4, eps=1, stoch_policy=True):
        self.tree = Tree()
        self.ANN = ANN
        self.c = c  # exploration constant
        self.eps = eps
        self.stoch_policy = stoch_policy

    def init_tree(self):
        self.tree.state_to_node.clear()

    def search(self, env, simulations_number):
        for i in range(simulations_number):
            simulation_env = env.sim_copy()
            self.simulate(simulation_env)
        D = self.tree.get_distribution(env)
        return D  # return visit distribution

    def simulate(self, env):
        traversed_nodes = self.sim_tree(env)
        if traversed_nodes:
            result = self.rollout(env)  # rollout env
            self.tree.backup(traversed_nodes, result)

    def sim_tree(self, env):
        path = []  # list of nodes traversed
        while not env.is_game_over():
            if not self.tree.get_node(env):
                return path
            path.append(self.tree.get_node(env))
            action = self.tree.tree_policy(env, self.c)  # find next action
            self.tree.get_node(env).set_last_action(action)
            env.move(action)
        return path

    def rollout(self, env):
        while not env.is_game_over():
            action = self.tree.rollout_policy(env, self.ANN, self.eps, self.stoch_policy)
            env.move(action)
        return env.result()
