import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class ANN:

    def __init__(self, alpha, epochs, io_layer_size, hidden_layer_sizes, activation_func, optimizer):
        self.alpha = alpha
        self.epochs = epochs
        # Build model of type Sequential()
        self.model = Sequential()
        self.model.add(Dense(io_layer_size, input_shape=(io_layer_size,)))
        for i in range(len(hidden_layer_sizes)):
            self.model.add(Dense(hidden_layer_sizes[i], activation=activation_func))
        self.model.add(Dense(io_layer_size, activation=tf.keras.activations.softmax))
        optimizer = self.__choose_optimizer(optimizer)
        self.model.compile(optimizer=optimizer,loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    def __choose_optimizer(self, optimizer):
        return {
            "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=self.alpha),
            "SGD": tf.keras.optimizers.SGD(learning_rate=self.alpha),
            "RMSProp": tf.keras.optimizers.RMSprop(learning_rate=self.alpha),
            "Adam": tf.keras.optimizers.Adam(learning_rate=self.alpha),
        }[optimizer]

    def fit(self, x, y):
        self.model.fit(x, y, epochs=self.epochs, verbose=1)

    def get_move(self, state, all_moves, possible_moves):
        probabilities = self.model.predict(state)[0]
        for i in range(len(probabilities)):
            if all_moves[i] not in possible_moves:
                probabilities[i] = 0
        sum_probs = sum(probabilities)
        probabilities = [p/sum_probs for p in probabilities]
        print(probabilities)
        return all_moves[np.argmax(probabilities)]


if __name__ == "__main__":
    from game import HexState
    from tree import Node
    from mcts import MonteCarloTreeSearch

    game = HexState(3)
    node = Node(game)
    model = ANN(0.001, 10, 9, [9,9,9], "relu", "Adagrad")
    #mcts = MonteCarloTreeSearch(node, model)
    model.fit(game.flat_state, np.array([[1/9]*9]))
    print(model.get_move(game.flat_state, game.all_moves, game.get_legal_actions()))
