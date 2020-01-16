import sys

class Hexboard:
    def __init__(self, type, size):
        self.size = size
        self.type = type
        if type == 'diamond':
            self.board = []
            for x in range(size):
                self.board.append([])
                for y in range(size):
                    self.board[x].append(Cell(1, self))
        elif type == 'triangle':
            self.board = []
            for x in range(size):
                self.board.append([])
                for y in range(size-x):
                    self.board[x].append(Cell(1, self))
        else:
            raise Exception('Boardtype '+ type +' not valid, choose either \'diamond\' or \'triangl\'')

    def get_cell(self, x,y):
        return self.board[x][y]

    def display(self):
        print(bcolors.PURPLE + "Boardtype: " + bcolors.WHITE + self.type.capitalize())
        print(bcolors.PURPLE + "Size: " + bcolors.WHITE + str(self.size)+'\n')
        n = self.size

        if self.type == 'triangle':
            k = 4*n-2
            for i in range(n):
                for j in range(k):
                    print(end=' ')
                k = k-3
                for j in range(i+1):
                    if self.board[-i-1][j].state == 1:
                        print(bcolors.YELLOW + '* ', end='    ')
                    if self.board[-i-1][j].state == 0:
                        print(bcolors.GREY + 'o ', end='    ')
                print('\n')

        if self.type == 'diamond':
            coordinates = []
            for x in range(n):
                for y in range(x+1):
                    coordinates.append((x-y, y))
            for c in coordinates[-1]:
                coordinates.append()

            k = 4*n-2
            for i in range(n):
                for j in range(k):
                    print(end=' ')
                k = k-2
                for j in range(i+1):
                    print('*', end='   ')
                print('\t')
            k = k*2-1
            for i in range(n-1,0,-1):
                for j in range(k-1,0,-1):
                    print(end=' ')
                k = k+2
                for j in range(i,0,-1):
                    print('*', end='   ')
                print('\t')

class Cell:
    # Cell on a game board, each state
    # representes whether the cell contains a piece or not
    def __init__(self, state, board):
        self.state = state
        self.board = board

    def update_state(self, new_state):
        self.state = new_state

class bcolors: #Class for setting colors in terminal output
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[0m'
    BOLD = '\033[1m'
    GREY = '\033[2m'
    BLACK = '\033[30m'
    DARK_RED = '\033[31m'
    CYAN = '\033[36m'

if __name__ == '__main__':
    b = Hexboard('triangle', 4)
    b.display()
