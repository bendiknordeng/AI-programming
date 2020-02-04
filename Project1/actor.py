from env import Board

class Actor:
    def __init__(self, agent, state, actions):
        self.agent = agent
        self.saps = self.generateSAP(state, actions)
        self.eligibilities = {}

    def generateSAP(self, state, actions):
        SAP = {}
        for move in actions:
            for to in actions[move]:
                SAP[(state,(move,to))] = 0
        return SAP

    def __toDec(n):
        sum = 0
        p = len(str(n))-1
        for i in str(n):
            if(int(i)):
                sum += 2**(int(i)*p)
            p-=1
        return sum

    def resetEligibilities(self):
        for key in self.eligibilities:
            self.eligibilities[key] = 0

    def chooseNext(self, moves):
        pass

    def score(self, currentState, nextState):
        pass

if __name__ == '__main__':
    pass
