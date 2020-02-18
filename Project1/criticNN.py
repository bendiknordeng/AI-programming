import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class CriticNN:

    def __init__(self, alpha, lam, gamma, hiddenLayerSizes, inputLayerSize):
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.eligibilities = []
        # Build model
        self.model = Sequential()
        self.model.add(Dense(inputLayerSize, activation='relu', input_dim=inputLayerSize))
        for i in range(len(hiddenLayerSizes)):
            self.model.add(Dense(hiddenLayerSizes[i], activation='relu'))
        self.model.add(Dense(1, activation='linear'))

        self.resetEligibilities()
        adagrad = tf.keras.optimizers.Adagrad(learning_rate=self.alpha)
        self.model.compile(optimizer=adagrad,loss=tf.keras.losses.MeanSquaredError(), run_eagerly = True)

    def resetEligibilities(self):
        self.eligibilities.clear()
        for params in self.model.trainable_weights:
            self.eligibilities.append(tf.zeros_like(params))

    def updateEligibilities(self):
        for i in range(len(self.eligibilities)):
            self.eligibilities[i] = self.lam * self.gamma * self.eligibilities[i]

    def valueState(self, state):
        state = [tf.strings.to_number(bin, out_type=tf.dtypes.int32) for bin in state]
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        return self.model(state).numpy()[0][0]

    def findTDError(self, reinforcement, lastState, state):
        target = reinforcement + self.gamma * self.valueState(state)
        td_error = target - self.valueState(lastState)
        return td_error

    def modify_gradients(self, gradients, td_error):
        for j in range(len(gradients)):
            gradients[j] = gradients[j] * 1/(2*td_error)
            self.eligibilities[j] = tf.add(self.eligibilities[j], gradients[j])
            gradients[j] = self.eligibilities[j] * td_error
        return gradients

    def fit(self, reinforcement, lastState, state, td_error):
        with tf.GradientTape() as tape:
            lastState, state, gamma, reinforcement = self.convertData(lastState, state, self.gamma, reinforcement)
            target = tf.add(reinforcement, tf.multiply(gamma, self.model(state)))
            prediction = self.model(lastState)
            loss = self.model.loss(target, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        modified_gradients = self.modify_gradients(gradients, td_error)
        self.model.optimizer.apply_gradients(zip(modified_gradients, self.model.trainable_variables))

    def convertData(self, lastState, state, gamma, reinforcement):
        lastState = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in lastState]  # convert to array
        lastState = tf.convert_to_tensor(np.expand_dims(lastState, axis=0))
        state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        gamma = tf.convert_to_tensor(self.gamma, dtype=tf.dtypes.float32)
        reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.dtypes.float32)
        return lastState, state, gamma, reinforcement
