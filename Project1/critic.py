import random

class Critic:
    def __init__(self, agent, state):
        self.agent = agent
        self.values = {}
        self.values[state] = random.random()
        self.eligibilities = {}

    def resetEligibilities(self):
        for key in self.eligibilities:
            self.eligibilities[key] = 0

    def update(state, state2, reward, action, action2):
        pass

if __name__ == '__main__':
    print()
