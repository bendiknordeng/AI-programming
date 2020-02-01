import networkx as nx
import matplotlib.pyplot as plt
import random
from cell import Cell


class Board:

    def __init__(self, type, size):
        self.type = type #0 is triangle, 1 is diamond.
        self.size = size #number of rows and columns of data structure of board.
        self.__cells = []
        self.__cellsWithPeg = []
        self.__emptyCells = []
        self.__edges = [] #format: tuple(), from cell with 1. lowest rowNumber and 2. lowest columnNumber
        self.__positions = {}
        self.__jumpedFrom = Cell(self, 0, 0)
        self.__jumpedTo = Cell(self, 0, 0)
        self.__jumpedFrom.setDummy()
        self.__jumpedTo.setDummy()
        self.__addCells()
        self.__positionCells()
        self.__addEdges()
        self.__G = nx.Graph()
        self.__G.add_nodes_from(self.__cellsWithPeg)
        self.__G.add_edges_from(self.__edges)

    def __addCells(self):
        if self.type == 0:  #if triangle: let column length be dynamic with k
            k = 1
            for r in range(self.size):
                for c in range(k):
                    self.__cellsWithPeg.append(Cell(self, r, c))
                k += 1
        elif self.type == 1:
            for r in range(self.size):
                for c in range(self.size):
                    self.__cellsWithPeg.append(Cell(self, r, c))

    def __positionCells(self): #dependent on that cells have been created
        if self.type == 0:
            for cell in self.__cellsWithPeg:
                r,c = cell.getRow(), cell.getColumn()
                self.__positions[cell] = (-10*r + 20*c, -10*r)
        elif self.type == 1:
            for cell in self.__cellsWithPeg:
                r,c = cell.getRow(), cell.getColumn()
                self.__positions[cell] = (-10*r + 10*c, -20*r - 20*c)

    def __addEdges(self): #dependent on that cells have been created.
        if self.type == 0:
            for n in range(len(self.__cellsWithPeg)):
                r,c = self.__cellsWithPeg[n].getRow(), self.__cellsWithPeg[n].getColumn()
                for m in range(n+1, len(self.__cellsWithPeg)):
                    i,j = self.__cellsWithPeg[m].getRow(), self.__cellsWithPeg[m].getColumn()
                    if i == r and j == c+1:
                        self.__edges.append((self.__cellsWithPeg[n],self.__cellsWithPeg[m]))
                    elif i == r+1 and j == c:
                        self.__edges.append((self.__cellsWithPeg[n],self.__cellsWithPeg[m]))
                    elif i == r+1 and j == c+1:
                        self.__edges.append((self.__cellsWithPeg[n],self.__cellsWithPeg[m]))
        elif self.type == 1:
            for n in range(len(self.__cellsWithPeg)):
                r,c = self.__cellsWithPeg[n].getRow(), self.__cellsWithPeg[n].getColumn()
                for m in range(n+1, len(self.__cellsWithPeg)):
                    i,j = self.__cellsWithPeg[m].getRow(), self.__cellsWithPeg[m].getColumn()
                    if i == r+1 and j == c-1:
                        self.__edges.append((self.__cellsWithPeg[n],self.__cellsWithPeg[m]))
                    elif i == r+1 and j == c:
                        self.__edges.append((self.__cellsWithPeg[n],self.__cellsWithPeg[m]))
                    elif i == r and j == c+1:
                        self.__edges.append((self.__cellsWithPeg[n],self.__cellsWithPeg[m]))

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

    def __removePeg(self, r, c):
        for cell in self.__cellsWithPeg:
            if r == cell.getRow() and c == cell.getColumn():
                cell.removePeg()
                self.__cellsWithPeg.remove(cell)
                self.__emptyCells.append(cell)

    def removeRandomPegs(self, numberOfPegs = 1):
        for i in range(numberOfPegs):
            k = random.randint(0, len(self.__cellsWithPeg)-1)
            self.__emptyCells.append(self.__cellsWithPeg[k])
            self.__cellsWithPeg.remove(self.__cellsWithPeg[k])

    def jumpPegFromOver(self, jumpFrom = (-1,-1), jumpOver = (-1,-1) ):
        r,c = jumpFrom
        i,j = jumpOver
        if r-i == 1: #determine endR, row where peg ends up
            endR = r-2
        elif i-r == 1:
            endR = r+2
        elif r == i:
            endR = r
        if c-j == 1: #determine endC, column where peg ends up
            endC = c-2
        elif j-c == 1:
            endC = c+2
        elif c == j:
            endC = c
        if r >= 0 and c >= 0:
            for cell in self.__cellsWithPeg:
                if cell.getRow() == r and cell.getColumn() == c:
                    cell.jumpedFrom()
                    if self.__jumpedFrom.getState() != -1:
                        self.__jumpedFrom.removePeg()
                        self.__emptyCells.append(self.__jumpedFrom)
                    self.__jumpedFrom = cell
                elif cell.getRow() == i and cell.getColumn() == j:
                    cellToRemove = cell
                    cell.removePeg()
                    self.__emptyCells.append(cell)
            for cell in self.__emptyCells:
                if cell.getRow() == endR and cell.getColumn() == endC:
                    cell.jumpedTo()
                    if self.__jumpedTo.getState() != -1:
                        self.__jumpedTo.placePeg()
                        self.__cellsWithPeg.append(self.__jumpedTo)
                    self.__jumpedTo = cell
            self.__cellsWithPeg.remove(self.__jumpedFrom)
            self.__cellsWithPeg.remove(cellToRemove)
            self.__emptyCells.remove(self.__jumpedTo)


def main():
    #board = Board(0,4)
    #board.draw()
    #board.removeRandomPegs(2)
    #board.draw()

main()
