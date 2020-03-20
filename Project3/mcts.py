from tree import Tree
import time

class MonteCarloTreeSearch:
    def __init__(self, ANN, c=1):
        self.tree = Tree()
        self.ANN = ANN
        self.eps = 1
        self.c = c  # exploration constant
        self.time_random_rollout = []
        self.time_ANN_rollout = []

    def init_tree(self):
        self.tree.state_to_node.clear()

    def search(self, env, simulations_number):
        for i in range(simulations_number):
            simulation_env = env.sim_copy()
            self.simulate(simulation_env)
        D = self.tree.get_distribution(env)
        return self.time_random_rollout, self.time_ANN_rollout, self.tree.tree_policy(env, 0), D  # find greedy best action

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
            start = time.time()
            ANN_used, action = self.tree.rollout_policy(env, self.ANN, self.eps)
            time_spent = time.time() - start
            if ANN_used:
                self.time_ANN_rollout.append(time_spent)
            else:
                self.time_random_rollout.append(time_spent)
            env.move(action)
        return env.result()
