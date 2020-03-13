from game import HexState
from mcts import MonteCarloTreeSearch
from tree import Node
from ANN import ANN
from tqdm import tqdm
import copy
import random

def RL_algorithm(episodes, simulations, training_batch_size, board_size, ann_save_interval, ANN, eps, eps_decay):
    cases = []
    for i in tqdm(range(episodes)):
        action = Node(HexState(board_size))
        while not action.is_terminal_node():
            action = MonteCarloTreeSearch(action, ANN, eps).best_action(simulations)
            visits_distribution = action.parent.get_normalized_visits(simulations)
            cases.append((action.parent.flat_game_state, visits_distribution))
        for case in random.sample(cases, min(len(cases),training_batch_size)):
            ANN.fit(case[0], case[1])
        if i % ann_save_interval == 0:
            ANN.model.save_weights('models/model_'+str(i)+'_simulations')
        eps *= eps_decay
