import torch
import numpy as np
import pdb
from tqdm import tqdm

class ANN:
    def __init__(self, io_dim, H_dims, learning_rate, optimizer, activation_fn, epochs):
        self.alpha = learning_rate
        self.epochs = epochs
        activation_fn = self.__choose_activation_fn(activation_fn)
        layers = [torch.nn.Linear(io_dim+1,H_dims[0])]
        layers.append(activation_fn) if activation_fn != None else None
        for i in range(len(H_dims)-1):
            layers.append(torch.nn.Linear(H_dims[i], H_dims[i+1]))
            layers.append(activation_fn) if activation_fn != None else None
        layers.append(torch.nn.Linear(H_dims[-1],io_dim))
        layers.append(torch.nn.Softmax(dim=-1))
        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.optimizer = self.__choose_optimizer(list(self.model.parameters()), optimizer)

    def fit(self, cases, debug=False):
        input = torch.tensor(cases[0]).float()
        target = torch.tensor(cases[1]).float()
        for i in tqdm(range(self.epochs)):
            pred = self.model(input)
            loss = self.loss_fn(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if debug: return float(loss.data.numpy())

    def get_loss(self, cases):
        input = torch.tensor(cases[0]).float()
        target = torch.tensor(cases[1]).float()
        pred = self.model(input)
        return self.loss_fn(pred, target).data.numpy()

    def make_dict(self, inputs, targets):
        input = torch.tensor(inputs).float()
        target = torch.tensor(targets).float()
        with torch.no_grad():
            pred = self.model(input)
        dict = {}
        inputs = input.data.numpy()
        targets = np.around(target.data.numpy()*100, decimals = 1)
        preds = np.around(pred.data.numpy()*100, decimals = 1)
        for j in range(len(target.data.numpy())):
            input, tar, pred = tuple(inputs[j]), targets[j], preds[j]
            if dict.get(input) == None:
                dict[tuple(input)] = [(tar, pred)]
            else:
                dict[tuple(input)].append((tar, pred))
        return dict


    def accuracy(self, cases, debug = False):
        pred = self.forward(cases[0])
        target = torch.tensor(cases[1]).float()
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
        probs = self.forward(env.flat_state).data
        factor = [1 if move in legal else 0 for move in env.all_moves]
        index = np.argmax([0 if not factor[i] else probs[i] for i in range(env.size**2)])
        return probs,env.all_moves[index]

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
