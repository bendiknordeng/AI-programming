from actor import Actor
from critic import Critic
from env import Board
import numpy as np

class Agent:
    def __init__(self, board, alphaActor, alphaCritic, eps, gamma):
        self.board = board
        self.alphaActor = alphaActor
        self.alphaCritic = alphaCritic
        self.eps = eps
        self.gamma = gamma
        #self.lastState
        #self.nextState
        #self.lastAction
        self.nextAction = None
        self.actor = Actor(self, self.alphaActor, self.eps, self.gamma)
        self.critic = Critic(self, self.alphaCritic, self.eps, self.gamma)


    def findAction(self):
        self.actor.createNewSAPs()
        self.nextAction = self.actor.getNextMove()

    def runEpisode(self):
        reward = 0
        while reward == 0:
            self.findAction()
            self.board.jumpPegFromTo(self.nextAction[0],self.nextAction[1])
            self.createCriticValues()
            self.updateCriticValues()
            reward = self.getReward()
        self.board.draw()
        maxSteps = 100

    def getState(self):
        return self.board.stringBoard()

    def getActions(self):
        return self.board.generateValidMoves()

    def getReward(self):
        return self.board.reward()

    def createCriticValues(self):
        self.critic.createNewValues()

    def updateCriticValues(self):
        self.critic.updateValues()
if __name__ == '__main__':

    alpha = 0.85
    eps = 0.9 #lambda
    gamma = 0.95

    board = Board(0, 3)
    board.removePegs([(0,0),(2,1),(2,0)]) #leaves no choices in moves.
    agent = Agent(board, alpha, alpha, eps, gamma)
    agent.runEpisode()
    """
    print(agent.getReward())
    board.draw()
    board.jumpPegFromTo((2,2),(0,0))
    board.draw()
    board.jumpPegFromTo((0,0),(2,0))
    board.draw()
    print(agent.getReward())
    print(agent.boardToString())
    agent.transition()
    board.draw()
    """
