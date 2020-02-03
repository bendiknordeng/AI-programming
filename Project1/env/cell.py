class Cell:
    def __init__(self, board):
        self.board = board #the board of cells
        self.state = 1 # States 0: empty, 1: peg,
                       # 2: peg jumped to cell, 3: peg jumped from cell, -1: dummy cell

    def setDummy(self):
        self.state = -1

    def removePeg(self):
        self.state = 0

    def placePeg(self):
        self.state = 1

    def jumpedTo(self):
        self.state = 2

    def jumpedFrom(self):
        self.state = 3

    def isEmpty(self):
        return self.state == 0 or self.state == 3

    def isDummy(self):
        return self.state == -1

    def getState(self):
        return self.state

    def __str__(self):
        pos = ''
        for key, cell in self.board.cells.items():
            if cell == self:
                pos = key
        return '\nPosition: ' + str(pos) + '\n' + 'State: ' + str(self.state)
