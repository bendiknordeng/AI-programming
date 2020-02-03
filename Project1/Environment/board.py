import networkx as nx
import matplotlib.pyplot as plt
import random
from cell import Cell


class Board:

    def __init__(self, type, size):
        self.type = type #0 is triangle, 1 is diamond.
        self.size = size #number of rows and columns of data structure of board.
        self.__cells = {} #key (row, column), value: Cell object
        self.__cellsWithPeg = []
        self.__emptyCells = []
        self.__edges = [] #format: tuple(), from cell with 1. lowest rowNumber and 2. lowest columnNumber
        self.__positions = {}
        self.__jumpedFrom = Cell(self)
        self.__jumpedTo = Cell(self)
        self.__jumpedFrom.setDummy()
        self.__jumpedTo.setDummy()
        self.__addCells()
        self.__positionCells()
        self.__addEdges()
        self.__G = nx.Graph()
        self.__G.add_nodes_from(self.__cellsWithPeg)
        self.__G.add_edges_from(self.__edges)

    def draw(self):
        nx.draw(self.__G, pos=self.__positions, nodelist=self.__emptyCells, node_color='black', node_size = 800)
        nx.draw(self.__G, pos=self.__positions, nodelist=self.__cellsWithPeg, node_color='blue', node_size = 800)
        if self.__jumpedTo.getState() != -1 and self.__jumpedFrom.getState() != -1:
            nx.draw(self.__G, pos=self.__positions, nodelist=[self.__jumpedTo], node_color='blue', node_size = 2400)
            nx.draw(self.__G, pos=self.__positions, nodelist=[self.__jumpedFrom], node_color='black', node_size = 200)
        #plt.savefig("board.png")
        plt.show()

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
            k = random.randint(0, len(self.__cellsWithPeg)-1)
            self.__emptyCells.append(self.__cellsWithPeg[k])
            self.__cellsWithPeg.remove(self.__cellsWithPeg[k])

    def jumpPegFromTo(self, jumpFrom = (-1,-1), jumpTo = (-1,-1)):
        rFrom, cFrom = jumpFrom
        rTo, cTo = jumpTo
        rOver, cOver = self.__findOverPos(rFrom, cFrom, rTo, cTo)
        if self.__isValidMove(rFrom, cFrom, rOver, cOver, rTo, cTo): #add validation function
            cellFrom = self.__cells[(rFrom, cFrom)]
            cellOver = self.__cells[(rOver, cOver)]
            cellTo = self.__cells[(rTo, cTo)]
            cellFrom.jumpedFrom()
            cellOver.removePeg()
            cellTo.jumpedTo()
            if not (self.__jumpedFrom.isDummy() and self.__jumpedTo.isDummy()):
                self.__jumpedFrom.removePeg()
                self.__jumpedTo.placePeg()
                self.__emptyCells.append(self.__jumpedFrom)
                self.__cellsWithPeg.append(self.__jumpedTo)
            self.__emptyCells.append(cellOver)
            self.__jumpedFrom = cellFrom
            self.__jumpedTo = cellTo
            self.__cellsWithPeg.remove(self.__jumpedFrom)
            self.__cellsWithPeg.remove(cellOver)
            self.__emptyCells.remove(self.__jumpedTo)
        else:
            print("The move is not valid") #NOT IMPLEMENTED

    def generateValidMoves(self): #should return dict of valid moves. Key,value pairs: (from), [(to)]
        validMoves = {}
        for (rFrom, cFrom) in self.__cells:
            toPositions = []
            for (rTo, cTo) in self.__cells:
                rOver, cOver = self.__findOverPos(rFrom, cFrom, rTo, cTo)
                if self.__isValidMove(rFrom, cFrom, rOver, cOver, rTo, cTo):
                    toPositions.append((rTo,cTo))
            if len(toPositions) > 0:
                validMoves[(rFrom, cFrom)] = toPositions
        return validMoves

    def __isValidMove(self, rFrom, cFrom, rOver, cOver, rTo, cTo):
        if rTo == None and cTo == None:
            return False
        fromValid = False
        overValid = False
        toValid = False
        for (r,c) in self.__cells:
            if r == rFrom and c == cFrom and not self.__cells[(r,c)].isEmpty():
                fromValid = True
            elif r == rOver and c == cOver and not self.__cells[(r,c)].isEmpty():
                overValid = True
            elif r == rTo and c == cTo and self.__cells[(r,c)].isEmpty():
                toValid = True
        return fromValid and overValid and toValid

    def __findOverPos(self, rFrom, cFrom, rTo, cTo):
        rOver, cOver = None, None
        if rFrom - rTo == 2: #determine rOver, row where peg ends up
            rOver = rFrom - 1
        elif rTo - rFrom == 2:
            rOver = rFrom + 1
        elif rFrom == cTo:
            rOver = rFrom
        if cFrom - cTo == 2: #determine cOver, column where peg ends up
            cOver = cFrom - 1
        elif cTo - cFrom == 2:
            cOver = cFrom + 1
        elif cFrom == cTo:
            cOver = cFrom
        return rOver, cOver

    def __removePeg(self, r, c):
        cell = self.__cells[(r,c)]
        cell.removePeg()
        self.__cellsWithPeg.remove(cell)
        self.__emptyCells.append(cell)

    def __addEdges(self): #dependent on that cells have been created.
        if self.type == 0:
            for (r,c) in self.__cells:
                for (i,j) in self.__cells:
                    if i == r and j == c+1:
                        self.__edges.append((self.__cells[(r,c)],self.__cells[(i,j)]))
                    elif i == r+1 and j == c:
                        self.__edges.append((self.__cells[(r,c)],self.__cells[(i,j)]))
                    elif i == r+1 and j == c+1:
                        self.__edges.append((self.__cells[(r,c)],self.__cells[(i,j)]))
        elif self.type == 1:
            for (r,c) in self.__cells:
                for (i,j) in self.__cells:
                    if i == r+1 and j == c-1:
                        self.__edges.append((self.__cells[(r,c)],self.__cells[(i,j)]))
                    elif i == r+1 and j == c:
                        self.__edges.append((self.__cells[(r,c)],self.__cells[(i,j)]))
                    elif i == r and j == c+1:
                        self.__edges.append((self.__cells[(r,c)],self.__cells[(i,j)]))

    def __positionCells(self): #dependent on that cells have been created
        if self.type == 0:
            for (r,c) in self.__cells:
                cell = self.__cells[(r,c)]
                self.__positions[cell] = (-10*r + 20*c, -10*r)
        elif self.type == 1:
            for (r,c) in self.__cells:
                cell = self.__cells[(r,c)]
                self.__positions[cell] = (-10*r + 10*c, -20*r - 20*c)

    def __addCells(self):
        if self.type == 0:  #if triangle: let column length be dynamic with k
            k = 1
            for r in range(self.size):
                for c in range(k):
                    cell = Cell(self)
                    self.__cellsWithPeg.append(cell)
                    self.__cells[(r,c)] = cell
                k += 1
        elif self.type == 1:
            for r in range(self.size):
                for c in range(self.size):
                    cell = Cell(self)
                    self.__cellsWithPeg.append(cell)
                    self.__cells[(r,c)] = cell

def main():
    #board = Board(0,4)
    #board.removePegs([(3,0),(3,1)])
    #dict = board.generateValidMoves()
    #for x in dict:
    #    print("jump from", x, "to")
    #    for y in dict[x]:
    #        print(y)
    #    print()
    #board.draw()
    #jumpFrom, jumpTo = (1,0), dict[(1,0)][0]
    #board.jumpPegFromTo(jumpFrom, jumpTo)
    #board.draw()

    #test diamond

    #board2 = Board(1,4)
    #board2.removePegs([(2,1), (2,3)])
    #board2.jumpPegFromTo((0,3),(2,3))
    #dict = board2.generateValidMoves()
    #for x in dict:
    #    print("jump from", x, "to")
    #    for y in dict[x]:
    #        print(y)
    #    print()
    #    print()
    #board2.draw()
    #board2.jumpPegFromTo((2,3),(2,1))
    #dict = board2.generateValidMoves()
    #for x in dict:
    #    print("jump from", x, "to")
    #    for y in dict[x]:
    #        print(y)
    #    print()
    #    print()
    #board2.draw()

main()
