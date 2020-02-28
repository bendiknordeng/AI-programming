from game import NIM, Ledge
import random
from tqdm import tqdm
from tree import Node


class MCTS:
    def __init__(self, G, M, c):
        self.G = G
        self.M = M
        self.c = c
        self.env = env

    #def play_games(self, game_mode, P, N, K, B, verbose):
    #    wins = 0
    #    run_list = range(self.G) if verbose else tqdm(range(self.G))
    #    for i in run_list:
    #        if game_mode == 0:
    #            self.env = NIM(P, N, K)
    #        else:
    #            self.env = Ledge(P, B)
    #        current = self.env.root
    #        if verbose:
    #            print("Initial state: {}".format(current.state))
    #        while self.__non_terminal(current):
    #            current = self.__tree_search(current)
    #            if verbose:
    #                self.env.print_move(current)
    #        if verbose:
    #            print("Player {} wins".format(1 if current.parent.turn else 2))
    #        if current.turn != self.env.root.turn:
    #            wins += 1
    #    print("Starting player wins {}/{} ({:.0f}%)".format(wins, self.G, 100 * wins / self.G))

    def tree_search(self):
        root = self.env.root
        for i in range(self.M):
            leaf = self.__traverse(root)  # leaf = unvisited node
            simulation_result = self.__rollout(leaf, root)
            self.__backpropagate(leaf, simulation_result)
        best_child = self.__best_child(root)
        best_child.reset()
        return best_child

    def __traverse(self, node):
        while self.__fully_expanded(node):
            print(node.children)
            node = self.__best_uct(node)
        if not node.children:
            self.env.generate_children(node)
        return self.__pick_unvisited(node.children) if self.__non_terminal(node) else node

    def __best_uct(self, node):
        best_value = float("-inf")
        best_nodes = []
        for child in node.children:
            if child.Q + child.u > best_value:
                best_value = child.Q + child.u
                best_nodes = [child]
            elif child.Q + child.u == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)

    def __fully_expanded(self, node):
        if not node.children:
            return False
        for child in node.children:
            if child.visits == 0:
                return False
        return True

    def __pick_unvisited(self, children):
        unvisited = []
        for child in children:
            if child.visits == 0:
                unvisited.append(child)
        return random.choice(unvisited)

    def __rollout(self, node, starting_player):
        while self.__non_terminal(node):
            action = random.choice(self.env.generate_valid_actions(node.state))
            new_state = self.env.next_state(node.state, action)
            node = Node(not node.turn, new_state, node, action)
        return self.env.get_reinforcement(node, starting_player)

    def __non_terminal(self, node):
        return not self.env.final_state(node)

    def __backpropagate(self, node, result):
        node.visits += 1
        while not node.is_root:
            node.update_values(result, self.c)
            node = node.parent

    def __best_child(self, node):
        max_visits = float("-inf")
        best = None
        for child in node.children:
            if child.visits > max_visits:
                max_visits = child.visits
                best = child
        return best


if __name__ == '__main__':
    G = 1  # number of games in batch
    M = 10  # number of rollouts per game move
    P = 1  # (1/2/3): Player 1 starts/Player 2 starts/Random player startsudfar
    c = 1  # exploration constant
    N = 10  # Inittial pile for NIM
    K = 3  # Max pieces for each action in NIM
    B = [1, 0, 0, 2]  # board for ledge
    game_mode = 0  # (0/1): NIM/Ledge

    env = NIM(P, N, K) if game_mode == 0 else Ledge(P, B)
    mcts = MCTS(G, M, c)

    action = mcts.tree_search()
    print(action.parent)
    print(action.prev_action)

    #verbose = True
    #mcts.play_games(game_mode, P, N, K, B, verbose)
