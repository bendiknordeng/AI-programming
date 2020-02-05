import networkx as nx
import matplotlib.pyplot as plt
import random
from env import Cell

class Board:
    def __init__(self, type, size):
        self.type = type #0 is triangle, 1 is diamond.
        self.size = size #number of rows and columns of data structure of board.
        self.cells = {} #key (row, column), value: Cell object
        self.cellsWithPeg = []
        self.emptyCells = []
        self.__edges = [] #format: tuple(), from cell with 1. lowest rowNumber and 2. lowest columnNumber
        self.positions = {}
        self.__jumpedFrom = Cell(self)
        self.__jumpedTo = Cell(self)
        self.__jumpedFrom.setDummy()
        self.__jumpedTo.setDummy()
        self.__addCells()
        self.__positionCells()
        self.__addEdges()
        self.__G = nx.Graph()
        self.__G.add_nodes_from(self.cellsWithPeg)
        self.__G.add_edges_from(self.__edges)

    def draw(self, pause = 0):
        fig = plt.figure(figsize =(9,7))
        plt.axes()
        nx.draw(self.__G, pos=self.positions, nodelist=self.emptyCells, node_color='black', node_size = 800, ax = fig.axes[0])
        nx.draw(self.__G, pos=self.positions, nodelist=self.cellsWithPeg, node_color='blue', node_size = 800, ax = fig.axes[0])
        if not self.__jumpedTo.isDummy() and not self.__jumpedFrom.isDummy():
            nx.draw(self.__G, pos=self.positions, nodelist=[self.__jumpedTo], node_color='blue', node_size = 2400, ax = fig.axes[0])
            nx.draw(self.__G, pos=self.positions, nodelist=[self.__jumpedFrom], node_color='black', node_size = 200, ax = fig.axes[0])
        if pause:
            plt.show(block = False)
            plt.pause(pause)
            plt.close()
        else:
            plt.show(block = True)

    def removePegs(self, positions = [(-1,-1)]):
        for pos in positions:
            r,c = pos
            if self.type == 0:
                if 0 <= r <= (self.size -1) and 0 <= c <= r:
                    self.__removePeg(r,c)
            elif self.type == 1:
                if 0 <= r <= (self.size-1) and 0 <= c <= (self.size-1):
                    self.__removePeg(r,c)

    def removeRandomPegs(self, numberOfPegs = 1):
        for i in range(numberOfPegs):
            k = random.randint(0, len(self.cellsWithPeg)-1)
            self.cellsWithPeg[k].removePeg()
            self.emptyCells.append(self.cellsWithPeg[k])
            self.cellsWithPeg.remove(self.cellsWithPeg[k])

    def jumpPegFromTo(self, jumpFrom = (-1,-1), jumpTo = (-1,-1)):
        rFrom, cFrom = jumpFrom
        rTo, cTo = jumpTo
        rOver, cOver = self.__findOverPos(rFrom, cFrom, rTo, cTo)
        if self.__isValidMove(rFrom, cFrom, rOver, cOver, rTo, cTo): #add validation function
            if not (self.__jumpedFrom.isDummy() and self.__jumpedTo.isDummy()):
                self.__jumpedFrom.removePeg()
                self.__jumpedTo.placePeg()
                self.emptyCells.append(self.__jumpedFrom)
                self.cellsWithPeg.append(self.__jumpedTo)
            cellFrom = self.cells[(rFrom, cFrom)]
            cellOver = self.cells[(rOver, cOver)]
            cellTo = self.cells[(rTo, cTo)]
            cellFrom.jumpedFrom()
            cellOver.removePeg()
            cellTo.jumpedTo()
            self.emptyCells.append(cellOver)
            self.__jumpedFrom = cellFrom
            self.__jumpedTo = cellTo
            self.cellsWithPeg.remove(self.__jumpedFrom)
            self.cellsWithPeg.remove(cellOver)
            self.emptyCells.remove(self.__jumpedTo)
        else:
            print("The move is not valid")

    def generateValidMoves(self): #should return dict of valid moves. Key,value pairs: (from), [(to)]
        validMoves = {}
        for (rFrom, cFrom) in self.cells:
            topositions = []
            for (rTo, cTo) in self.cells:
                rOver, cOver = self.__findOverPos(rFrom, cFrom, rTo, cTo)
                if self.__isValidMove(rFrom, cFrom, rOver, cOver, rTo, cTo):
                    topositions.append((rTo,cTo))
            if len(topositions) > 0:
                validMoves[(rFrom, cFrom)] = topositions
        return validMoves

    def reinforcement(self):
        numberOfPegs = 0
        for pos in self.cells:
            if not self.cells[pos].isEmpty():
                numberOfPegs += 1

        if numberOfPegs ==1:
            return 10
        elif len(self.generateValidMoves()) <= 0:
            return -10
        else:
            return 0

    def stringBoard(self):
        state = ''
        for pos in self.cells:
            if self.cells[pos].isEmpty():
                state += '0'
            else:
                state += '1'
        return state

    def __isValidMove(self, rFrom, cFrom, rOver, cOver, rTo, cTo):
        if rFrom == None or cFrom == None or rOver ==None or cOver==None or rTo==None or cTo == None:
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
            if r == rFrom and c == cFrom and not self.cells[(r,c)].isEmpty():
                fromValid = True
            elif r == rOver and c == cOver and not self.cells[(r,c)].isEmpty():
                overValid = True
            elif r == rTo and c == cTo and self.cells[(r,c)].isEmpty():
                toValid = True
        return fromValid and overValid and toValid

    def __findOverPos(self, rFrom, cFrom, rTo, cTo):
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
        return rOver, cOver

    def __removePeg(self, r, c):
        cell = self.cells[(r,c)]
        cell.removePeg()
        self.cellsWithPeg.remove(cell)
        self.emptyCells.append(cell)

    def __addEdges(self): #dependent on that cells have been created.
        if self.type == 0:
            for (r,c) in self.cells:
                for (i,j) in self.cells:
                    if i == r and j == c+1:
                        self.__edges.append((self.cells[(r,c)],self.cells[(i,j)]))
                    elif i == r+1 and j == c:
                        self.__edges.append((self.cells[(r,c)],self.cells[(i,j)]))
                    elif i == r+1 and j == c+1:
                        self.__edges.append((self.cells[(r,c)],self.cells[(i,j)]))
        elif self.type == 1:
            for (r,c) in self.cells:
                for (i,j) in self.cells:
                    if i == r+1 and j == c-1:
                        self.__edges.append((self.cells[(r,c)],self.cells[(i,j)]))
                    elif i == r+1 and j == c:
                        self.__edges.append((self.cells[(r,c)],self.cells[(i,j)]))
                    elif i == r and j == c+1:
                        self.__edges.append((self.cells[(r,c)],self.cells[(i,j)]))

    def __positionCells(self): #dependent on that cells have been created
        if self.type == 0:
            for (r,c) in self.cells:
                cell = self.cells[(r,c)]
                self.positions[cell] = (-10*r + 20*c, -10*r)
        elif self.type == 1:
            for (r,c) in self.cells:
                cell = self.cells[(r,c)]
                self.positions[cell] = (-10*r + 10*c, -20*r - 20*c)

    def __addCells(self):
        if self.type == 0:  #if triangle: let column length be dynamic with k
            k = 1
            for r in range(self.size):
                for c in range(k):
                    cell = Cell(self)
                    self.cellsWithPeg.append(cell)
                    self.cells[(r,c)] = cell
                k += 1
        elif self.type == 1:
            for r in range(self.size):
                for c in range(self.size):
                    cell = Cell(self)
                    self.cellsWithPeg.append(cell)
                    self.cells[(r,c)] = cell

if __name__ == '__main__':
    pass
