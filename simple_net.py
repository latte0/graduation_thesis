import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


def gauss(x, a = 1, mu = 0, sigma = 1):
    return a * torch.exp(-(x - mu)**2 / (2*sigma**2))

    

class Net(nn.Module):




    def __init__(self, Y, X):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.Y = Y
        self.train_X = X
        self.h = nn.Parameter(torch.tensor(1.0, requires_grad = True))

    def leave_one_out(self, X, Xw):
        numerator = 0
        denominator = 0
        result = []
        print("h")
        print(self.h)
        for j, x_j in enumerate(X):
                tmp = gauss( (( torch.mv(self.fc1.weight , x_j) - Xw ) / self.h ))
                denominator += tmp
                numerator += tmp * self.Y[j]
                #print(tmp)
                #print(self.Y[j])
                #print(tmp * self.Y[j])

        g = numerator/denominator
        return g

                
    def forward(self, x):
        xw = self.fc1(x)
        #reluかleave_one_out切り分け
        y = self.leave_one_out(self.train_X, xw)
        #y = F.relu(xw)
        return y

iris = datasets.load_iris()
y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
y[np.arange(len(iris.target)), iris.target] = 1
X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.95)
x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)
x_static = Variable(torch.from_numpy(X_train).float(), requires_grad=False)
#print(x)
y = Variable(torch.from_numpy(y_train).float())
net = Net(y, x_static)
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(3000):
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    print(loss)
    loss.backward()
    optimizer.step()

    # test
    outputs = net(Variable(torch.from_numpy(X_test).float()))
    _, predicted = torch.max(outputs.data, 1)
    y_predicted = predicted.numpy()
    y_true = np.argmax(y_test, axis=1)
    accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    print('accuracy: {0}%'.format(accuracy))


# utility function to predict for an unknown data
def predict(X):
    X = Variable(torch.from_numpy(np.array(X)).float())
    outputs = net(X)
    return np.argmax(outputs.data.numpy())