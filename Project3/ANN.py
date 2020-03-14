import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class ANN:
    def __init__(self, io_layer_size, hidden_layer_sizes=[], weights=None, alpha=0.01, epochs=10, activation_func="relu", optimizer="Adagrad"):
        self.alpha = alpha
        self.epochs = epochs
        # Build model of type Sequential()
        self.model = Sequential()
        self.model.add(Dense(io_layer_size, input_dim=io_layer_size))
        for i in range(len(hidden_layer_sizes)):
            self.model.add(Dense(hidden_layer_sizes[i], activation=activation_func))
        self.model.add(Dense(io_layer_size, activation=tf.keras.activations.softmax))
        optimizer = self.__choose_optimizer(optimizer)
        self.model.compile(optimizer=optimizer,loss=tf.keras.losses.MeanSquaredError())
        if weights:
            self.model.load_weights(weights)

    def __choose_optimizer(self, optimizer):
        return {
            "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=self.alpha),
            "SGD": tf.keras.optimizers.SGD(learning_rate=self.alpha),
            "RMSProp": tf.keras.optimizers.RMSprop(learning_rate=self.alpha),
            "Adam": tf.keras.optimizers.Adam(learning_rate=self.alpha),
        }[optimizer]

    def get_move(self, state):
        possible_moves = state.get_legal_actions()
        all_moves = state.all_moves
        probabilities = self.model(state.flat_state).numpy()[0]
        for i in range(len(probabilities)):
            if all_moves[i] not in possible_moves:
                probabilities[i] = 0
        sum_probs = sum(probabilities)
        probabilities = [p/sum_probs for p in probabilities]
        return all_moves[np.argmax(probabilities)]


    def fit(self, input, target):
        self.model.fit(input,target, epochs = 10, verbose = 0)

    #def fit(self, input, target):
    #    with tf.GradientTape() as tape:
    #        pred = self.model(input)
    #        loss = self.model.loss(target, pred)
    #    grads = tape.gradient(loss, self.model.trainable_variables)
    #    self.model.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))


if __name__ == "__main__":
    from game import HexState
    from tree import Node
    from mcts import MonteCarloTreeSearch

    game = HexState(4)
    node = Node(game)
    model = ANN(0.001, 10, 16, [9,9,9], "relu", "Adagrad")
    #mcts = MonteCarloTreeSearch(node, model)
    model.fit(game.flat_state, np.array([[1/9]*16]))
    print(model.get_move(game))
