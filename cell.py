class Cell:



    def __init__(self, board, row, column):
        self.board = board #the board of cells
        self.row = row #row position in board
        self.column = column #column position in board
        self.state = 0 # States 0: empty, 1: peg, 2: most recently hopped peg, 3: peg jumped from here

    def clearCell(self):
        self.state = 0

    def placePeg(self):
        self.state = 1

    def jumpedToSelf(self):
        self.state = 2

    def jumpedFromSelf(self):
        self.state = 3

    def getRow(self):
        return self.row

    def getColumn(self):
        return self.column
