import sys
sys.path.append('../')
from env import Board

class Actor:
    def __init__(self, learningRate = 0.1, discount = 0.9):
        self.learningRate = learningRate
        self.discount = discount

    def chooseNext(self, moves):
        pass

    def score(self, currentState, nextState):
        pass

if __name__ == '__main__':
    board = Board(0,4)
    a = Actor(board, display = True)
