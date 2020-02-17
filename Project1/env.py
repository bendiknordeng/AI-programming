import networkx as nx
import matplotlib.pyplot as plt
import random

class Board:
    def __init__(self, type, size, initial = [], random = 0):
        self.type = type #0 is triangle, 1 is diamond.
        self.size = size #number of rows and columns of data structure of board.
        self.cells = {} #key (row, column), value: status
        self.__edges = [] #format: tuple(), from cell with 1. lowest rowNumber and 2. lowest columnNumber
        self.positions = {}
        self.jumpedFrom = None
        self.jumpedTo = None
        self.__addCells()
        self.__positionCells()
        self.__addEdges()
        self.initial = initial
        if random > 0:
            self.getRandomPegs(random)

        if len(self.initial) > 0:
                self.removePegs(self.initial)

        self.__G = nx.Graph()
        self.__G.add_nodes_from(self.cellsWithPeg())
        self.__G.add_edges_from(self.__edges)

    def draw(self, pause = 0):
        fig = plt.figure(figsize = (9,7))
        plt.axes()
        filled = self.cellsWithPeg()
        empty = self.emptyCells()
        if not (self.jumpedTo is None and self.jumpedFrom is None):
            filled.remove(self.jumpedTo)
            empty.remove(self.jumpedFrom)
            nx.draw(self.__G, pos = self.positions, nodelist = [self.jumpedTo], node_color='blue', node_size = 2400, ax = fig.axes[0])
            nx.draw(self.__G, pos = self.positions, nodelist = [self.jumpedFrom], node_color='black', node_size = 200, ax = fig.axes[0])
        nx.draw(self.__G, pos = self.positions, nodelist = filled, node_color='blue', node_size = 800, ax = fig.axes[0])
        nx.draw(self.__G, pos = self.positions, nodelist = empty, node_color='black', node_size = 800, ax = fig.axes[0])
        if pause:
            plt.show(block = False)
            plt.pause(pause)
            plt.close()
        else:
            plt.show(block = True)

    def reset(self):
        for key in self.cells:
            self.cells[key] = 1
        self.jumpedFrom = None
        self.jumpedTo = None

        if len(self.initial) > 0:
            self.removePegs(self.initial)

    def emptyCells(self):
        positions = []
        for pos in self.cells:
            if self.__cellEmpty(pos):
                positions.append(pos) # empty cells
        return positions

    def cellsWithPeg(self):
        positions = []
        for pos in self.cells:
            if not self.__cellEmpty(pos):
                positions.append(pos) # empty cells
        return positions

    def removePegs(self, positions = [(-1,-1)]):
        for pos in positions:
            r,c = pos
            if self.type == 0:
                if 0 <= r <= (self.size -1) and 0 <= c <= r:
                    self.__removePeg(r,c)
            elif self.type == 1:
                if 0 <= r <= (self.size-1) and 0 <= c <= (self.size-1):
                    self.__removePeg(r,c)

    def getRandomPegs(self, numberOfPegs = 1):
        keys = list(self.cells)
        for key in self.initial:
            keys.remove(key)
        for i in range(numberOfPegs):
            k = random.randint(0, len(keys)-1)
            self.initial.append(keys.pop(k))

    def execute(self, action):
        jumpFrom, jumpTo = action
        jumpOver = self.__findOverPos(jumpFrom, jumpTo)
        if self.__isValidAction(jumpFrom, jumpOver, jumpTo):
            self.cells[jumpFrom] = 0
            self.cells[jumpOver] = 0
            self.cells[jumpTo] = 1
            self.jumpedFrom = jumpFrom
            self.jumpedTo = jumpTo
            return True
        else:
            print("Invalid move: {} => {} => {} \nValid: {}".format(jumpFrom, jumpOver, jumpTo,(self.__isValidAction(jumpFrom, jumpOver, jumpTo))))
            return False

    def generateActions(self): #should return dict of valid moves. Key,value pairs: (from), [(to)]
        validMoves = {}
        for jumpFrom in self.cells:
            topositions = []
            for jumpTo in self.cells:
                jumpOver = self.__findOverPos(jumpFrom, jumpTo)
                if self.__isValidAction(jumpFrom, jumpOver, jumpTo):
                    topositions.append(jumpTo)
            if len(topositions) > 0:
                validMoves[jumpFrom] = topositions
        return validMoves

    def numberOfPegsLeft(self):
        numberOfPegs = 0
        for pos in self.cells:
            if not self.__cellEmpty(pos):
                numberOfPegs += 1
        return numberOfPegs

    def reinforcement(self):
        if self.numberOfPegsLeft() == 1:
            return 1
        elif len(self.generateActions()) <= 0:
            return -1
        else:
            return 0

    def getState(self):
        state = ''
        for pos in self.cells:
            if self.__cellEmpty(pos):
                state += '0'
            else:
                state += '1'
        return state

    def __isValidAction(self, jumpFrom, jumpOver, jumpTo):
        rFrom, cFrom = jumpFrom
        rOver, cOver = jumpOver
        rTo, cTo = jumpTo
        if rFrom == None or cFrom == None or rOver == None or cOver== None or rTo== None or cTo == None:
            return False
        if self.type == 0:
            if rFrom - rOver == -1 and cFrom - cOver == 1:
                return False
            elif rOver - rFrom == -1 and cOver - cFrom == 1:
                return False
            elif rFrom - rOver == 2 or rOver - rFrom == 2 : #catch case moving vertically
                return False
        if self.type == 1:
            if rOver - rFrom == 1 and cOver - cFrom == 1:  #catch case moving vertically
                return False
            elif rFrom -rOver == 1 and cFrom - cOver == 1:
                return False
        fromValid = False
        overValid = False
        toValid = False
        for (r,c) in self.cells:
            if r == rFrom and c == cFrom and not self.__cellEmpty((r, c)):
                fromValid = True
            elif r == rOver and c == cOver and not self.__cellEmpty((r, c)):
                overValid = True
            elif r == rTo and c == cTo and self.__cellEmpty((r, c)):
                toValid = True
        return fromValid and overValid and toValid

    def __cellEmpty(self, pos):
        return self.cells[pos] == 0

    def __findOverPos(self, jumpFrom, jumpTo):
        rFrom, cFrom = jumpFrom
        rTo, cTo = jumpTo
        rOver, cOver = None, None
        if rFrom - rTo == 2: #determine rOver, row where peg ends up
            rOver = rFrom - 1
        elif rTo - rFrom == 2:
            rOver = rFrom + 1
        elif rFrom == rTo:
            rOver = rFrom
        if cFrom - cTo == 2: #determine cOver, column where peg ends up
            cOver = cFrom - 1
        elif cTo - cFrom == 2:
            cOver = cFrom + 1
        elif cFrom == cTo:
            cOver = cFrom
        return (rOver, cOver)

    def __removePeg(self, r, c):
        self.cells[(r,c)] = 0

    def __addEdges(self): #dependent on that cells have been created.
        if self.type == 0:
            for (r,c) in self.cells:
                for (i,j) in self.cells:
                    if i == r and j == c+1:
                        self.__edges.append(((r,c),(i,j)))
                    elif i == r+1 and j == c:
                        self.__edges.append(((r,c),(i,j)))
                    elif i == r+1 and j == c+1:
                        self.__edges.append(((r,c),(i,j)))
        elif self.type == 1:
            for (r,c) in self.cells:
                for (i,j) in self.cells:
                    if i == r+1 and j == c-1:
                        self.__edges.append(((r,c),(i,j)))
                    elif i == r+1 and j == c:
                        self.__edges.append(((r,c),(i,j)))
                    elif i == r and j == c+1:
                        self.__edges.append(((r,c),(i,j)))

    def __positionCells(self): #dependent on that cells have been created
        if self.type == 0:
            for (r,c) in self.cells:
                self.positions[(r,c)] = (-10*r + 20*c, -10*r)
        elif self.type == 1:
            for (r,c) in self.cells:
                self.positions[(r,c)] = (-10*r + 10*c, -20*r - 20*c)

    def __addCells(self):
        if self.type == 0:  #if triangle: let column length be dynamic with r
            for r in range(self.size):
                for c in range(r+1):
                    self.cells[(r,c)] = 1 #place peg in pos (r,c)
        elif self.type == 1:
            for r in range(self.size):
                for c in range(self.size):
                    self.cells[(r,c)] = 1 #place peg in pos (r,c)

if __name__ == '__main__':
    board = Board(0,4)
    board.removePegs([(1,1)])
    board.draw()
    board.jumpPegFromTo((3,3),(1,1))

    board.draw()
