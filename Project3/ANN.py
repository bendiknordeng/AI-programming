import torch
import numpy as np
import pdb
from tqdm import tqdm

class ANN:
    def __init__(self, io_dim, H_dims, learning_rate, optimizer, activation_fn, epochs):
        self.alpha = learning_rate
        self.epochs = epochs
        activation_fn = self.__choose_activation_fn(activation_fn)
        layers = self.gen_layers(io_dim, H_dims, activation_fn)
        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.optimizer = self.__choose_optimizer(list(self.model.parameters()), optimizer)

    def gen_layers(self, io_dim, H_dims, activation_fn):
        layers = [torch.nn.Linear(io_dim*2+2,H_dims[0])]
        layers.append(activation_fn) if activation_fn != None else None
        for i in range(len(H_dims)-1):
            layers.append(torch.nn.Linear(H_dims[i], H_dims[i+1]))
            layers.append(activation_fn) if activation_fn != None else None
        layers.append(torch.nn.Linear(H_dims[-1],io_dim))
        layers.append(torch.nn.Softmax(dim=-1))
        return layers

    def fit(self, input, target, debug=False):
        input = torch.tensor(input).float()
        target = torch.tensor(target).float()
        for i in range(self.epochs):
            pred = self.model(input)
            loss = self.loss_fn(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if debug: return float(loss.data.numpy())

    def get_loss(self, input, target):
        target = torch.tensor(target).float()
        pred = self.forward(input)
        loss = self.loss_fn(pred, target).data.numpy()
        return loss

    def accuracy(self, input, target, debug = False):
        target = torch.tensor(target).float()
        pred = self.forward(input)
        pred_indices = torch.argmax(pred, 1)
        target_indices = torch.argmax(target, 1)
        eq_sum = torch.sum(torch.eq(pred_indices, target_indices))
        acc = float((eq_sum/float(len(pred_indices))).data.numpy())
        if debug: pdb.set_trace()
        return acc

    def forward(self, input):
        with torch.no_grad():
            return self.model(torch.tensor(input).float())

    def get_move(self, env):
        legal = env.get_legal_actions()
        factor = [1 if move in legal else 0 for move in env.all_moves]
        probs = self.forward(env.flat_state).data.numpy()
        sum = 0
        new_probs = np.zeros(env.size ** 2)
        for i in range(env.size ** 2):
            if factor[i]:
                sum += probs[i]
                new_probs[i] = probs[i]
            else:
                new_probs[i] = 0
        new_probs /= sum
        indices = np.arange(env.size ** 2)
        stoch_index = np.random.choice(indices, p=new_probs)
        greedy_index = np.argmax(new_probs)
        return new_probs, stoch_index, greedy_index

    def save(self, size, level):
        torch.save(self.model, "models/{}_ANN_level_{}".format(size,level))
        print("Model has been saved to models/{}_ANN_level_{}".format(size,level))

    def load(self, size, level):
        self.model = torch.load("models_stoch/{}_ANN_level_{}".format(size,level))
        print("Loaded model from models_stoch/{}_ANN_level_{}".format(size,level))

    def __choose_optimizer(self, params, optimizer):
        return {
            "Adagrad": torch.optim.Adagrad(params, lr=self.alpha),
            "SGD": torch.optim.SGD(params, lr=self.alpha),
            "RMSprop": torch.optim.RMSprop(params, lr=self.alpha),
            "Adam": torch.optim.Adam(params, lr=self.alpha)
        }[optimizer]

    def __choose_activation_fn(self, activation_fn):
        return {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
            "linear": None
        }[activation_fn]
