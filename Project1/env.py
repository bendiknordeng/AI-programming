import networkx as nx
import matplotlib.pyplot as plt
import random

class Board:
    def __init__(self, type, size, initial = [], random = 0):
        self.__type = type # 0 is triangle, 1 is diamond.
        self.__size = size # number of rows and columns of data structure of board.
        self.__cells = {} # key (row, column), value: peg status
        self.__edges = [] # format: tuples with position of two different neighbouring cells
        self.__positions = {} # dictionary to define placements for visualization
        self.__jumpedFrom = None # hold position of last cell that is jumped from
        self.__jumpedTo = None  # hold position of last cell that is jumped to
        # __initialize cells, positions and edges, clear cells, and create networkx graph for visualization
        self.__addCells()
        self.__positionCells()
        self.__addEdges()
        self.__initial = initial # list of inital cells that is cleared in initial state
        if random > 0:
            self.__getRandomPegs(random)
        if len(self.__initial) > 0:
            self.__removePegs(self.__initial)
        self.__G = nx.Graph()
        self.__G.add_nodes_from(self.__cellsWithPeg())
        self.__G.add_edges_from(self.__edges)

    # draw with break between frames given in "play"
    def draw(self, animation_delay = 0):
        fig = plt.figure(figsize = (9,7))
        plt.axes()
        filled = self.__cellsWithPeg()
        empty = self.__emptyCells()
        if not (self.__jumpedTo is None and self.__jumpedFrom is None): # if first move has been made
            filled.remove(self.__jumpedTo)
            empty.remove(self.__jumpedFrom)
            nx.draw(self.__G, pos = self.__positions, nodelist = [self.__jumpedTo], node_color='blue', node_size = 2400, ax = fig.axes[0])
            nx.draw(self.__G, pos = self.__positions, nodelist = [self.__jumpedFrom], node_color='black', node_size = 200, ax = fig.axes[0])
        nx.draw(self.__G, pos = self.__positions, nodelist = filled, node_color='blue', node_size = 800, ax = fig.axes[0])
        nx.draw(self.__G, pos = self.__positions, nodelist = empty, node_color='black', node_size = 800, ax = fig.axes[0])
        if animation_delay: # run animation automatically if delay > 0
            plt.show(block = False)
            plt.pause(animation_delay)
            plt.close()
        else: # show single figure if delay not given
            plt.show(block = True)

    # reset board to the initial state
    def reset(self):
        for key in self.__cells:
            self.__cells[key] = 1
        self.__jumpedFrom = None
        self.__jumpedTo = None

        if len(self.__initial) > 0:
            self.__removePegs(self.__initial)
        return self.getState(), self.__generateActions()

    # perform action on board
    def execute(self, action):
        lastState = self.getState() #save for pass to agent
        jumpFrom, jumpTo = action
        jumpOver = self.__findOverPos(jumpFrom, jumpTo)
        if self.__isValidAction(jumpFrom, jumpOver, jumpTo):
            self.__cells[jumpFrom] = 0
            self.__cells[jumpOver] = 0
            self.__cells[jumpTo] = 1
            self.__jumpedFrom = jumpFrom
            self.__jumpedTo = jumpTo
        else: # give error message if invalid move
            print("Invalid move: {} => {} => {} \nValid: {}".format(jumpFrom, jumpOver, jumpTo,(self.__isValidAction(jumpFrom, jumpOver, jumpTo))))
        return lastState, self.getState(), self.__reinforcement(), self.__generateActions()

    def numberOfPegsLeft(self):
        numberOfPegs = 0
        for pos in self.__cells:
            if not self.__cellEmpty(pos):
                numberOfPegs += 1
        return numberOfPegs

    # return string representing boardstate
    def getState(self):
        state = ''
        for pos in self.__cells:
            if self.__cellEmpty(pos):
                state += '0'
            else:
                state += '1'
        return state

    # return positions of empty cells
    def __emptyCells(self):
        positions = []
        for pos in self.__cells:
            if self.__cellEmpty(pos):
                positions.append(pos) # empty cells
        return positions

    # return positions of non-empty cells
    def __cellsWithPeg(self):
        positions = []
        for pos in self.__cells:
            if not self.__cellEmpty(pos):
                positions.append(pos) # empty cells
        return positions

    # removes all pegs given their positions
    def __removePegs(self, positions = [(-1,-1)]):
        for pos in positions:
            r,c = pos
            if self.__type == 0:
                if 0 <= r <= (self.__size -1) and 0 <= c <= r: # check if position is valid
                    self.__removePeg(r,c)
            elif self.__type == 1:
                if 0 <= r <= (self.__size-1) and 0 <= c <= (self.__size-1): # check if position is valid
                    self.__removePeg(r,c)

    # removes random pegs and adds them to __initial state
    def __getRandomPegs(self, numberOfPegs = 1):
        keys = list(self.__cells)
        for key in self.__initial:
            keys.remove(key)
        for i in range(numberOfPegs):
            k = random.randint(0, len(keys)-1)
            self.__initial.append(keys.pop(k))

    # return dict of valid moves. Key,value pairs: (from), [(to)]
    def __generateActions(self):
        validMoves = {}
        for jumpFrom in self.__cells:
            topositions = []
            for jumpTo in self.__cells:
                jumpOver = self.__findOverPos(jumpFrom, jumpTo) # retrieve peg jumped over
                if self.__isValidAction(jumpFrom, jumpOver, jumpTo):
                    topositions.append(jumpTo)
            if len(topositions) > 0:
                validMoves[jumpFrom] = topositions
        return validMoves

    def __reinforcement(self):
        if self.numberOfPegsLeft() == 1: # won
            return 100
        elif len(self.__generateActions()) <= 0: # no valid actions, lost state
            return -100
        else:
            return 0

    # validates if the jump from, over and to cell is a valid jump
    def __isValidAction(self, jumpFrom, jumpOver, jumpTo):
        rFrom, cFrom = jumpFrom
        rOver, cOver = jumpOver
        rTo, cTo = jumpTo
        if rFrom == None or cFrom == None or rOver == None or cOver== None or rTo== None or cTo == None:
            return False
        if self.__type == 0: # catch illegal jumps in triangle
            if rFrom - rOver == -1 and cFrom - cOver == 1:
                return False
            elif rOver - rFrom == -1 and cOver - cFrom == 1:
                return False
            elif rFrom - rOver == 2 or rOver - rFrom == 2 : # case moving vertically
                return False
        if self.__type == 1: # catch illegal jumps in triangle
            if rOver - rFrom == 1 and cOver - cFrom == 1:  # case moving vertically
                return False
            elif rFrom -rOver == 1 and cFrom - cOver == 1:
                return False
        fromValid = False
        overValid = False
        toValid = False
        for (r,c) in self.__cells:
            if r == rFrom and c == cFrom and not self.__cellEmpty((r, c)):
                fromValid = True
            elif r == rOver and c == cOver and not self.__cellEmpty((r, c)):
                overValid = True
            elif r == rTo and c == cTo and self.__cellEmpty((r, c)):
                toValid = True
        return fromValid and overValid and toValid

    def __cellEmpty(self, pos):
        return self.__cells[pos] == 0

    def __findOverPos(self, jumpFrom, jumpTo):
        rFrom, cFrom = jumpFrom
        rTo, cTo = jumpTo
        rOver, cOver = None, None
        if rFrom - rTo == 2: # determine row to jump over
            rOver = rFrom - 1
        elif rTo - rFrom == 2:
            rOver = rFrom + 1
        elif rFrom == rTo:
            rOver = rFrom
        if cFrom - cTo == 2: # determine column to jump over
            cOver = cFrom - 1
        elif cTo - cFrom == 2:
            cOver = cFrom + 1
        elif cFrom == cTo:
            cOver = cFrom
        return (rOver, cOver)

    def __removePeg(self, r, c):
        self.__cells[(r,c)] = 0

    def __addEdges(self): #dependent on that cells have been initialized
        if self.__type == 0:
            for (r,c) in self.__cells:
                for (i,j) in self.__cells:
                    if i == r and j == c+1:
                        self.__edges.append(((r,c),(i,j)))
                    elif i == r+1 and j == c:
                        self.__edges.append(((r,c),(i,j)))
                    elif i == r+1 and j == c+1:
                        self.__edges.append(((r,c),(i,j)))
        elif self.__type == 1:
            for (r,c) in self.__cells:
                for (i,j) in self.__cells:
                    if i == r+1 and j == c-1:
                        self.__edges.append(((r,c),(i,j)))
                    elif i == r+1 and j == c:
                        self.__edges.append(((r,c),(i,j)))
                    elif i == r and j == c+1:
                        self.__edges.append(((r,c),(i,j)))

    def __positionCells(self): #dependent on that cells have been initialized
        if self.__type == 0:
            for (r,c) in self.__cells:
                self.__positions[(r,c)] = (-10*r + 20*c, -10*r)
        elif self.__type == 1:
            for (r,c) in self.__cells:
                self.__positions[(r,c)] = (-10*r + 10*c, -20*r - 20*c)

    def __addCells(self):
        if self.__type == 0:
            for r in range(self.__size):
                for c in range(r+1): # let column length be dynamic with r for triangle
                    self.__cells[(r,c)] = 1 # place peg in pos (r,c)
        elif self.__type == 1:
            for r in range(self.__size):
                for c in range(self.__size):
                    self.__cells[(r,c)] = 1 # place peg in pos (r,c)
