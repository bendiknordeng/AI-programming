import random

class CriticTable:
    def __init__(self, alpha, lam, gamma):
        self.__alpha = alpha
        self.__lam = lam
        self.__gamma = gamma
        self.__values = {}
        self.__eligs = {}  # key, value is state, eligibility

    # initialize state values with small, random numbers
    def createStateValues(self, state):
        if self.__values.get(state) == None:
            self.__values[state] = random.random()

    def findTDError(self, reinforcement, lastState, state):
        self.td_error = reinforcement + self.__gamma * self.__values[state] - self.__values[lastState]
        return self.td_error

    # update all state values in current trace
    def updateStateValues(self):
        for state in self.__eligs:
            self.__values[state] = self.__values[state] + self.__alpha * self.td_error * self.__eligs[state]

    # create eligibility wiht value 1 for last visited state
    def createEligibility(self, state):
        if self.__eligs.get(state) == None:
            self.__eligs[state] = 1

    # decay eligibilities with factor gamma*lambda
    def updateEligibilities(self):
        for state in self.__eligs:
            self.__eligs[state] = self.__gamma * self.__lam * self.__eligs[state]

    def resetEligibilities(self):
        self.__eligs.clear() # remove all eligibilities to make new trace

    # return critic's valuation of input state
    def stateValue(self, state):
        return self.__values[state]
