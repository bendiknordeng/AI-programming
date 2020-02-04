from actor import Actor
from critic import Critic
from env import Board
import numpy as np

class Agent:
    def __init__(self, board, initialActions):
        self.initialState = self.boardToBinary(board)
        self.initialActions = initialActions
        self.actor = Actor(self, self.initialState, self.initialActions)
        self.critic = Critic(self, self.initialState)

    def runEpisode(self):
        alpha = 0.85
        epsilon = 0.9
        gamma = 0.95
        maxSteps = 100
        self.actor.resetEligibilities()
        self.critic.resetEligibilities()
        # actor.state = initialState
        # actor.saps = actor.generateSAP(initialState, initialActions)
        # critic.state = initialState
        action = self.actor.chooseNext(self.state)


    def boardToBinary(self, board):
        state = ""
        for pos in board.cells:
            if board.cells[pos].isEmpty():
                state += '0'
            else:
                state += '1'
        return int(state)

if __name__ == '__main__':
    # epsilon = 0.9
    # total_episodes = 10000
    # max_steps = 100
    # alpha = 0.85
    # gamma = 0.95

    board = Board(0, 5)
    board.removeRandomPegs(2)
    initialActions = board.generateValidMoves()

    agent = Agent(board, initialActions)
    print(agent.actor.saps)
