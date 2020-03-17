from tree import Tree
import copy


class MonteCarloTreeSearch:
    def __init__(self, c=1):
        self.tree = Tree()
        self.c = c  # exploration constant

    def init_tree(self, env):
        self.tree.state_to_node.clear()

    def search(self, env, simulations_number):
        for i in range(simulations_number):
            simulation_env = copy.copy(env)
            self.simulate(simulation_env)
        return self.tree.tree_policy(env, 0)  # find greedy best action

    def simulate(self, env):
        traversed_nodes = self.sim_tree(env)
        z = self.rollout(env)  # rollout env
        self.tree.backup(traversed_nodes, z)

    def sim_tree(self, env):
        c = self.c
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
            action = self.tree.rollout_policy(env)
            env.move(action)
        return env.result()
