import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pdb

class CriticNN:

    def __init__(self, alpha, lam, gamma, inputDim=0, nodesInLayers=[0]):
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.surprise = 0
        self.eligibilities = []
        self.model = Sequential()
        self.model.add(
            Dense(nodesInLayers[0], activation='relu', input_dim=inputDim))
        for i in range(1, len(nodesInLayers)):
            self.model.add(Dense(nodesInLayers[i], activation='relu'))


        self.model.add(Dense(1, activation = tf.keras.layers.LeakyReLU(alpha=0.05)))
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

    def modify_gradients(self, gradients, loss, td_error,i,prediction):
        alpha = tf.convert_to_tensor(self.alpha, dtype=tf.dtypes.float32)
        #if prediction.numpy()[0][0] > 5:
            #pdb.set_trace()

        #print("td_error",td_error)
        for j in range(len(gradients)):
            #if j == 1:
            #    print(self.eligibilities[j].numpy())
            #    print()
            self.eligibilities[j] = tf.add(self.eligibilities[j], gradients[j])
            #if j == 1:
            #    print(self.eligibilities[j].numpy())
            #    print()
            #if j == 1:
            #    print(gradients[j].numpy())
            #    print()
            gradients[j] = self.eligibilities[j] * td_error
            #if j == 1:
            #    print(gradients[j].numpy())
            #    print()
            #print()
        return gradients

    def gen_loss(self, stringLastState, lastState, td_error):
        prediction = self.model(lastState, training = True)
        target = tf.add(td_error, prediction)
        loss = self.loss_fn(target, prediction)
        return loss

    def fit(self, reinforcement, lastState, state, td_error,i, verbose=False):
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
        #print("reinforcement",reinforcement.numpy(), "loss", loss.numpy(), "laststate prediction", lastStateString, prediction.numpy(), "state target", stateString, target.numpy())
        gradients = self.modify_gradients(gradients, loss, td_error, i ,prediction)
        gradients, _ = tf.clip_by_global_norm(firstGradients, 1.0)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if verbose: self.display_useful_stuff()

    # Use the 'metric' to run a quick test on any set of features and targets.  A typical metric is some form of
    # 'accuracy', such as 'categorical_accuracy'.  Read up on Keras.metrics !!
    # Note that the model.metrics__names slot includes the name of the loss function (as 0th entry),
    # whereas the model.metrics slot does not include the loss function, hence the index+1 in the final line.
    # Use your debugger and go through the long list of slots for a keras model.  There are a lot of useful things
    # that you have access to.

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

    def gen_evaluation(self, features, target, avg=False, index=0):
        predictions = self.model(features)
        evaluation = self.model.metrics[index](targets, predictions)
        #  Note that this returns both a tensor (or value) and the NAME of the metric
        return (tf.reduce_mean(evaluation).numpy() if avg else evaluation, self.model.metrics_names[index + 1])

    def status_display(self, features, targets, mode='Train'):
        print(mode + ' *** ', end='')
        print('Loss: ', self.gen_loss(features, targets, avg=True), end=' : ')
        val, name = self.gen_evaluation(features, targets)
        print('Eval({0}): {1} '.format(name, val))

    def end_of_epoch_display(self, train_ins, train_targs, val_ins, val_targs):
        self.status_display(train_ins, train_targs, mode='Train')
        if len(val_ins) > 0:
            self.status_display(val_ins, val_targs, mode='      Validation')
