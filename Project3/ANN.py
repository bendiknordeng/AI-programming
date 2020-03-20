import torch
import numpy as np
import time
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
        layers.append(torch.nn.Softmax(dim = 0))
        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.optimizer = self.__choose_optimizer(list(self.model.parameters()), optimizer)

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
        float_tensor = torch.tensor(input).float()
        #start = time.time()
        #pred = self.model(float_tensor)
        #w_grad = time.time()-start
        #start = time.time()
        with torch.no_grad():
            pred = self.model(float_tensor)
        #no_grad = time.time()-start

        #print("prediction wi_grad", w_grad)
        #print("prediction no_grad", no_grad)
        #print("with_grad / no_grad", w_grad/no_grad)
        #print()
        return pred


    def get_move(self, env):
        legal = env.get_legal_actions()
        probs = self.forward(env.flat_state).data
        factor = [1 if move in legal else 0 for move in env.all_moves]
        index = np.argmax([0 if not factor[i] else probs[i] for i in range(env.size**2)])
        return env.all_moves[index]

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
