import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class CriticNN:

    def __init__(self, alpha, lam, gamma, inputDim=0, nodesInLayers=0):
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.eligibilities = []
        self.model = Sequential()
        # Create model with one hidden layer (Tesauro’s TD-Gammon system)
        self.model.add(Dense(inputDim, activation=tf.keras.layers.LeakyReLU(alpha=0.5), input_dim=inputDim))
        self.model.add(Dense(nodesInLayers, activation=tf.keras.layers.LeakyReLU(alpha=0.5)))
        self.model.add(Dense(1, activation = 'linear'))
        self.resetEligibilities()
        sgd = tf.optimizers.SGD(lr=alpha,momentum=0.9, nesterov=True)# decay=1e-6,
        #sgd = tf.optimizers.SGD(lr=alpha, momentum = 0.2)#, clipnorm = 1.0)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=sgd, loss=tf.keras.losses.MeanSquaredError(), run_eagerly = True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

        self.winningStates = []
        self.loosingStates = []
        self.episodeStates = []

    def resetEligibilities(self):
        self.eligibilities.clear()
        for params in self.model.trainable_weights:
            self.eligibilities.append(tf.zeros_like(params))

    def updateEligibilities(self):
        lambdaGamma = tf.convert_to_tensor(self.lam*self.gamma, dtype=tf.dtypes.float32)
        for i in range(len(self.eligibilities)):
            self.eligibilities[i] = tf.multiply(lambdaGamma, self.eligibilities[i])

    def valueState(self, state):
        state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        return self.model(state)

    def findTDError(self, reinforcement, lastState, state):
        target = reinforcement + self.gamma * self.valueState(state)
        td_error = target - self.valueState(lastState)
        return td_error

    def modify_gradients(self, gradients, loss, td_error):
        alpha = tf.convert_to_tensor(self.alpha, dtype=tf.dtypes.float32)
        for j in range(len(gradients)):
            self.eligibilities[j] = tf.add(self.eligibilities[j], gradients[j])
            gradients[j] = self.eligibilities[j] * td_error
        return gradients

    def fit(self, reinforcement, lastState, state, td_error):
        with tf.GradientTape() as tape:
            td_error = tf.convert_to_tensor(td_error, dtype=tf.dtypes.float32)
            lastState, state, gamma, reinforcement = self.convertData(lastState, state, self.gamma, reinforcement)
            target = tf.add(reinforcement, tf.multiply(gamma, self.model(state, training = True)))
            prediction = self.model(lastState, training = True)
            loss = self.loss_fn(target, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        modified_gradients = self.modify_gradients(gradients, loss, td_error)
        self.model.optimizer.apply_gradients(zip(modified_gradients, self.model.trainable_variables))

    def addEpisodeStates(self, to):
        episodeStates = tuple(self.episodeStates)
        if to == 1:
            if episodeStates in self.winningStates:
                self.winningStates.remove(episodeStates)
                self.winningStates.insert(0, episodeStates)
            if episodeStates not in self.winningStates:
                self.winningStates.insert(0, episodeStates)
        if to == -1:
            if episodeStates in self.loosingStates:
                self.loosingStates.remove(episodeStates)
                self.loosingStates.insert(0, episodeStates)
            if episodeStates not in self.loosingStates:
                self.loosingStates.insert(0, episodeStates)
        self.episodeStates.clear()

    def convertData(self, lastState, state, gamma, reinforcement):
        lastState = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in lastState]  # convert to array
        lastState = tf.convert_to_tensor(np.expand_dims(lastState, axis=0))
        state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        gamma = tf.convert_to_tensor(self.gamma, dtype=tf.dtypes.float32)
        reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.dtypes.float32)
        return lastState, state, gamma, reinforcement

    def printStateValues(self):
        print()
        print("winningStates")
        for path in self.winningStates:
            path = list(path)
            for state in path:
                stringState = state
                state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
                state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
                print(stringState, self.model(state, training = False).numpy())
            print()
        print()
        print("loosingStates")
        for path in self.loosingStates:
            path = list(path)
            for state in path:
                stringState = state
                state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
                state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
                print(stringState, self.model(state, training = False).numpy())
            print()
