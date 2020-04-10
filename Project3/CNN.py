import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game import HexGame

class CNN(nn.Module):
    def __init__(self, size, alpha=0.01, epochs=10, activation='ReLU', optimizer='Adam'):
        super(CNN, self).__init__()
        self.env = HexGame(size)
        self.size = size
        self.alpha = alpha
        self.epochs = epochs
        self.model = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(8,32,3),
            nn.ReLU(),
            nn.Conv2d(32,32,2),
            nn.ReLU(),
            nn.Conv2d(32,1,2),
            nn.ReLU(),
            nn.Conv2d(1,1,1),
            nn.Softmax(dim=1))
        params = list(self.parameters())
        self.optimizer = self.__choose_optimizer(params, optimizer)
        self.loss_fn = nn.BCELoss()

    def forward(self, x, training=False):
        self.train(training)
        x = self.transform_input(x)
        return self.model(x)

    def fit(self, x, y):
        y = torch.FloatTensor(y)
        for i in range(self.epochs):
            pred_y = self.forward(x, training=True)
            loss = self.loss_fn(pred_y, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        acc = pred_y.argmax(dim=1).eq(y.argmax(dim=1)).sum().numpy()/len(y)
        return loss.item(), acc

    def get_status(self, input, target):
        y = torch.FloatTensor(target)
        pred_y = self.forward(input)
        loss = self.loss_fn(pred_y, y)
        acc = pred_y.argmax(dim=1).eq(y.argmax(dim=1)).sum().numpy()/len(y)
        return loss.item(), acc

    def transform_input(self, x):
        '''
        Transforms flat game state into 9 input planes (size, size):
        - Empty/p1/p2       0/1/2   (empty/red/black)
        - To play           3/4     (p1/p2 to play)
        - P1 bridge         5       (red bridge endpoints)
        - P2 bridge         6       (black bridge endpoints)
        - To play bridge    7       (active if cell is a form bridge)
        - To play bridge    8       (active if cell is a save bridge) # TODO
        '''
        player = x[0]
        x = x[1:].reshape(self.size, self.size)
        planes = np.zeros(8*self.size**2).reshape(8,self.size,self.size)
        planes[player+2] += 1
        for r in range(self.size):
            for c in range(self.size):
                piece = x[r][c]
                planes[piece][r][c] = 1
                if (r, c) in self.env.bridge_neighbors:
                    for (rb, cb) in self.env.bridge_neighbors[(r,c)]:
                        if piece == 0:
                            if x[rb][cb] == player:
                                planes[7][r][c] = 1
                        else:
                            if x[rb][cb] == piece:
                                planes[piece+4][r][c] = 1
        planes = torch.FloatTensor(planes)
        import pdb; pdb.set_trace()
        return planes

    def get_move(self, env):
        legal = env.get_legal_actions()
        factor = [1 if move in legal else 0 for move in env.all_moves]
        probs = self.forward(env.flat_state).data.numpy()[0]
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
        torch.save(self.state_dict(), "models/{}_CNN_level_{}".format(size,level))
        print("Model has been saved to models/{}_CNN_level_{}".format(size,level))

    def load(self, size, level):
        self.load_state_dict(torch.load("models/{}_CNN_level_{}".format(size,level)))
        print("Loaded model from models/{}_CNN_level_{}".format(size,level))

    def __choose_optimizer(self, params, optimizer):
        return {
            "Adagrad": torch.optim.Adagrad(params, lr=self.alpha),
            "SGD": torch.optim.SGD(params, lr=self.alpha),
            "RMSprop": torch.optim.RMSprop(params, lr=self.alpha),
            "Adam": torch.optim.Adam(params, lr=self.alpha)
        }[optimizer]

    def __choose_activation_fn(self, activation_fn):
        return {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
        }[activation_fn]

if __name__ == '__main__':
    from game import HexGame
    env = HexGame(4)
    reds = [(1, 2),(1,1),(2,2),(3,0)]
    blacks = [(3,3),(0,0),(1,3),(0,2)]
    for cell in reds:
        env.state[cell] = 1
    for cell in blacks:
        env.state[cell] = 2
    CNN = CNN(4)
    CNN.transform_input(env.flat_state)
