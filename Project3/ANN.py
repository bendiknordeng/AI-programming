import torch
import numpy as np
from collections import OrderedDict

class ANN:
    def __init__(self, io_dim, H_dims, learning_rate, optimizer, activation, epochs):
        self.alpha = learning_rate
        self.epochs = epochs

        activation_fn = {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
            "linear": None
        }[activation]

        layers = [torch.nn.Linear(io_dim+1,H_dims[0])]
        if activation_fn != None:
            layers.append(activation_fn)
        for i in range(len(H_dims)-1):
            layers.append(torch.nn.Linear(H_dims[i], H_dims[i+1]))
            if activation_fn != None:
                layers.append(activation_fn)
        layers.append(torch.nn.Linear(H_dims[-1],io_dim))
        layers.append(torch.nn.Softmax())

        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.optimizer = self.__choose_optimizer(list(self.model.parameters()), optimizer)

    def __choose_optimizer(self, params, optimizer):
        return {
            "Adagrad": torch.optim.Adagrad(params, lr=self.alpha),
            "SGD": torch.optim.SGD(params, lr=self.alpha),
            "RMSprop": torch.optim.RMSprop(params, lr=self.alpha),
            "Adam": torch.optim.Adam(params, lr=self.alpha),
        }[optimizer]

    def fit(self, cases):
        input = torch.tensor([case[0] for case in cases]).float()
        target = torch.tensor([case[1] for case in cases]).float()
        for i in range(self.epochs):
            pred = self.model(input)
            loss = self.loss_fn(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, input):
        return self.model(torch.tensor(input).float())

if __name__ == "__main__":
    from game import HexGame
    from mcts import MonteCarloTreeSearch
    size = 3
    game = HexGame(size)
    ann = ANN(io_dim=size*size, H_dims=[10,10,10], learning_rate=1e-2, optimizer="Adagrad", activation="tanh", epochs=500)
    print(ann.model)

    input = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    target = [0, 0, 0.2, 0, 0.6, 0, 0.2, 0, 0]
