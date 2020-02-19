import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class CriticNN:

    def __init__(self, alpha, lam, gamma, hiddenLayerSizes, inputLayerSize):
        self.__alpha = alpha
        self.__lam = lam
        self.__gamma = gamma
        self.__eligibilities = []
        # Build model of type Sequential()
        self.__model = Sequential()
        self.__model.add(Dense(inputLayerSize, activation='relu', input_dim=inputLayerSize))
        for i in range(len(hiddenLayerSizes)):
            self.__model.add(Dense(hiddenLayerSizes[i], activation='relu'))
        self.__model.add(Dense(1))

        self.resetEligibilities()
        adagrad = tf.keras.optimizers.Adagrad(learning_rate=self.__alpha)
        self.__model.compile(optimizer=adagrad,loss=tf.keras.losses.MeanSquaredError(), run_eagerly = True)

    # generate eligibilities with equal shape as trainable weights
    def resetEligibilities(self):
        self.__eligibilities.clear()
        for params in self.__model.trainable_variables:
            self.__eligibilities.append(tf.zeros_like(params))

    # decay eligibilities with factor gamma*lambda
    def updateEligibilities(self):
        for i in range(len(self.__eligibilities)):
            self.__eligibilities[i] = self.__lam * self.__gamma * self.__eligibilities[i]

    # return the model's current valuation of given state
    def stateValue(self, state):
        state = [tf.strings.to_number(bin, out_type=tf.dtypes.int32) for bin in state] # convert to array
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        return self.__model(state).numpy()[0][0]

    # given the model's valuation of lastState and state, return td_error
    def findTDError(self, reinforcement, lastState, state):
        target = reinforcement + self.__gamma * self.stateValue(state)
        td_error = target - self.stateValue(lastState)
        return td_error

    # caluclate loss and apply modified gradients to weights via model optimizer
    def fit(self, reinforcement, lastState, state, td_error):
        with tf.GradientTape() as tape:
            lastState, state, gamma, reinforcement = self.__convertData(lastState, state, self.__gamma, reinforcement)
            target = tf.add(reinforcement, tf.multiply(gamma, self.__model(state)))
            prediction = self.__model(lastState)
            loss = self.__model.loss(target, prediction)
        gradients = tape.gradient(loss, self.__model.trainable_variables)
        modified_gradients = self.__modify_gradients(gradients, td_error)
        self.__model.optimizer.apply_gradients(zip(modified_gradients, self.__model.trainable_variables))

    def __modify_gradients(self, gradients, td_error):
        for j in range(len(gradients)):
            gradients[j] = gradients[j] * 1/(2*td_error) # retrieve gradient value of state w.r.t. weights
            self.__eligibilities[j] = tf.add(self.__eligibilities[j], gradients[j]) # adjust eligibilities with current gradient
            gradients[j] = self.__eligibilities[j] * td_error # gradients to be applied by optimizer
        return gradients
    # convert input to tensors
    def __convertData(self, lastState, state, gamma, reinforcement):
        lastState = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in lastState]  # convert to array
        lastState = tf.convert_to_tensor(np.expand_dims(lastState, axis=0))
        state = [tf.strings.to_number(bin, out_type=tf.dtypes.float32) for bin in state]
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        gamma = tf.convert_to_tensor(self.__gamma, dtype=tf.dtypes.float32)
        reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.dtypes.float32)
        return lastState, state, gamma, reinforcement
