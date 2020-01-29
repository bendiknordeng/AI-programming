import networkx as nx
import matplotlib.pyplot as plt
from cell import Cell


class Board:

    def __init__(self, type, size):
        self.type = type #0 is triangle, 1 is diamond.
        self.size = size #number of rows and columns of data structure of board.
        self.cells = []
        self.edges = [] #format: tuple(), from cell with 1. lowest rowNumber and 2. lowest columnNumber
        self.positions = {}

    def positionCells(self):
        if self.type == 0:
            for cell in self.cells:
                r,c = cell.getRow(), cell.getColumn()
                self.positions[cell] = (-10*r + 20*c, -10*r)

    def addCells(self):
        if self.type == 0:  #if triangle: let column length be dynamic with k
            k = 1
            for r in range(self.size):
                for c in range(k):
                    self.cells.append(Cell(self, r, c)) #add cell with position
                    #print("Added cell with row " + str(r) + " and column " + str(c))
                k += 1

    def addEdges(self): #must have made self.cells in a double array
        if self.type == 0:
            for n in range(len(self.cells)):
                #print("n: ", n)
                r,c = self.cells[n].getRow(), self.cells[n].getColumn()
                for m in range(n+1, len(self.cells)):
                    #print("m: ", m)
                    i,j = self.cells[m].getRow(), self.cells[m].getColumn()
                    if i == r and j == c+1:
                        self.edges.append((self.cells[n],self.cells[m]))
                        #print("Added edge from (" + str(r)+","+str(c) + ") to (" + str(i)+","+str(j)+")")
                    elif i == r+1 and j == c:
                        self.edges.append((self.cells[n],self.cells[m]))
                        #print("Added edge from (" + str(r)+","+str(c) + ") to (" + str(i)+","+str(j)+")")
                    elif i == r+1 and j == c+1:
                        self.edges.append((self.cells[n],self.cells[m]))



    def draw(self):
        jumpedFromCell = []
        jumpedToCell = []
        emptyCells = []
        cellsWithPeg= []

        for cell in self.cells:
            print(cell.getState())
            if cell.getState() == 0:
                emptyCells.append(cell)
            elif cell.getState() == 1:
                cellsWithPeg.append(cell)
            elif cell.getState() == 2:
                jumpedToCell.append(cell)
            else:
                jumpedFromCell.append(cell)

        G = nx.Graph()
        G.add_nodes_from(self.cells)
        G.add_edges_from(self.edges)

        print(emptyCells)
        nx.draw(G, pos=self.positions, nodelist=emptyCells, node_color='black', node_size = 1000)
        nx.draw(G, pos=self.positions, nodelist=cellsWithPeg, node_color='blue', node_size = 1000)
        nx.draw(G, pos=self.positions, nodelist=jumpedToCell, node_color='blue', node_size = 2500)
        nx.draw(G, pos=self.positions, nodelist=jumpedFromCell, node_color='black', node_size = 200)

        plt.savefig("board.png")
        plt.show()

    def removePeg(self, r, c):
        for cell in self.cells:
            if r == cell.getRow() and c == cell.getColumn():
                print("peg removed at " +str(r)+ "," + str(c))
                cell.removePeg()
                return

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
            for cell in self.cells:
                if cell.getRow() == r and cell.getColumn() == c:
                    fromCell = cell
                    print("fromCell: ", fromCell)
                elif cell.getRow() == i and cell.getColumn() == j:
                    overCell = cell
                    print("overCell: ", overCell)
                elif cell.getRow() == endR and cell.getColumn() == endC:
                    toCell = cell
                    print("toCell: ", toCell)

        if not fromCell.isEmpty() and not overCell.isEmpty() and toCell.isEmpty():
            fromCell.jumpedFrom()
            overCell.removePeg()
            toCell.jumpedTo()
            print("jumped from ", jumpFrom, " over ", jumpOver, " to ", (endR,endC))
        else:
            print("Not a valid jump")

def main():
    board = Board(0,4)
    board.addCells()
    board.addEdges()
    board.positionCells()
    board.draw()
    board.removePeg(3,2)
    board.draw()
    board.jumpPegFromOver((3,0),(3,1))
    board.draw()

main()
