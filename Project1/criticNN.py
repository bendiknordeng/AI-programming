import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class CriticNN:

    def __init__(self, alpha, lam, gamma, inputDim=0, nodesInLayers=[0]):
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.surprise = 0
        self.eligibilities = []
        self.model = Sequential()
        self.model.add(
            Dense(nodesInLayers[0], activation=tf.keras.layers.LeakyReLU(alpha=0.5), input_dim=inputDim))
        for i in range(1, len(nodesInLayers)):
            self.model.add(Dense(nodesInLayers[i], activation=tf.keras.layers.LeakyReLU(alpha=0.5)))

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

    def findTDError(self, reinforcement, lastState, state):
        # converting states from string to tensor
        lastState = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in lastState]  # convert to array
        state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
        lastState = tf.convert_to_tensor(np.expand_dims(lastState, axis=0))
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        gamma = tf.convert_to_tensor(self.gamma, dtype=tf.dtypes.float32)
        reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.dtypes.float32)
        td_error = tf.subtract(tf.add(reinforcement, tf.multiply(gamma, self.model(state))),self.model(lastState)).numpy()[0][0]
        return td_error

    def modify_gradients(self, gradients, loss, td_error):
        alpha = tf.convert_to_tensor(self.alpha, dtype=tf.dtypes.float32)
        for j in range(len(gradients)):
            self.eligibilities[j] = tf.add(self.eligibilities[j], gradients[j])
            gradients[j] = self.eligibilities[j] * td_error
        return gradients

    def fit(self, reinforcement, lastState, state, td_error, verbose=False):
        stateString = state
        lastStateString = lastState
        self.episodeStates.append(lastState)

        if reinforcement != 0:
            self.addEpisodeStates(reinforcement)

        with tf.GradientTape() as tape:
            lastState = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in lastState]  # convert to array
            lastState = tf.convert_to_tensor(np.expand_dims(lastState, axis=0))
            state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
            state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
            gamma = tf.convert_to_tensor(self.gamma, dtype=tf.dtypes.float32)
            reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.dtypes.float32)
            target = tf.add(reinforcement, tf.multiply(gamma, self.model(state, training = True)))
            prediction = self.model(lastState, training = True)
            loss = self.loss_fn(target, prediction)
        firstGradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(firstGradients, 1.0)
        gradients = self.modify_gradients(gradients, loss, td_error)
        gradients, _ = tf.clip_by_global_norm(firstGradients, 1.0)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if verbose: self.display_useful_stuff()

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

    def modelPred(self,state):
        state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        return self.model(state,training=False).numpy()[0][0]

    def display_useful_stuff(self):
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
