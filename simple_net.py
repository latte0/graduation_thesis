import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.datasets import load_digits
# pytorchのガウス関数

DATA_OUTPUT_LENGTH = 4

def gauss(x, a=1, mu=0, sigma=1):
    return a * torch.exp(-(x - mu)**2 / (2*sigma**2))

def create_list_data(p):
    l = []
    for _ in range(0, DATA_OUTPUT_LENGTH):
        l.append(p)
    return l


loss_list = []
acc_list = []


class Net(nn.Module):

    def __init__(self, Y, calc_Y, X, calc_X, settings):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        # leave_ont_outのために事前に入力と出力をセットしておく
        self.Y = Y
        self.calc_Y = calc_Y
        self.train_X = X
        self.calc_X = calc_X
        self.settings = settings
        self.test = False
        # バンド幅も推定する
        self.h = nn.Parameter(torch.tensor(1.5, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

    # leave_one_out推定量の計算

    def leave_one_out(self, Xw):
        numerator = 0
        denominator = 0
        result = []
        # print("h")
        #print(self.h)
        for j, x_j in enumerate(self.train_X):
            #print(torch.mv(self.fc1.weight, x_j) - Xw)
            tmp = gauss(((self.fc1(x_j) - Xw) / self.h))
            # print(len(Xw))
            tmp[j] = 0
            denominator += tmp
            numerator += tmp * self.Y[j]

        g = numerator/denominator
        return g

    def set_test(self, test):
        self.test = test



    def leave_one_out_output(self, Xw):
        numerator = 0
        denominator = 0
        result = []
        # print("h")
        for j, x_j in enumerate(self.calc_X):
            tmp = gauss(((self.fc1(x_j) - Xw) / self.h))
            denominator += tmp
            numerator += tmp * self.calc_Y[j]

        g = numerator/denominator
        return g

    def forward(self, x):

        # reluかleave_one_out切り分け
        if self.settings["activation"] == "leave_one_out":
            if(not self.test):
                y = self.leave_one_out(self.fc1(x))
            if(self.test):
                y = self.leave_one_out_output(self.fc1(x))
        else:
            xw = self.fc1(x)
            y = self.sigmoid(xw)
        return y


# データの用意
iris = datasets.load_iris()
y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
y[np.arange(len(iris.target)), iris.target] = 1
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, y, test_size=0.1)


X_calc, _, y_calc, _ = train_test_split(
    iris.data, y, test_size=0.90)




x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)

# leave_one_outのために定数として一応用意しておく
x_static = Variable(torch.from_numpy(X_train).float(), requires_grad=False)
x_static_calc = Variable(torch.from_numpy(X_calc).float(), requires_grad=False)


y = Variable(torch.from_numpy(y_train).float())
y_calc = Variable(torch.from_numpy(y_calc).float())

# leave one outの計算のため、事前に入力と出力のパラメータをセットしておく
net = Net(y, y_calc, x_static, x_static_calc, {"activation": "leave_one_out"})
optimizer = optim.SGD(net.parameters(), lr=1.01)
criterion = nn.MSELoss()


test_input_x = np.linspace(-20, 20, 200)
test_input_x_list = []

for p in test_input_x:
    test_input_x_list.append([p, p, p])

test_input_x_torch = torch.from_numpy(np.array(test_input_x_list)).float()

plt.ion()

for i in range(3000):
    net.set_test(True)
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    print(i)
    print(loss)
    loss.backward()
    optimizer.step()

    loss_list.append(loss)

    '''
    if(i % 100 == 0):
        test_input_y_torch = net.leave_one_out_output(test_input_x_torch)
        test_input_y = test_input_y_torch.to('cpu').detach().numpy().copy()

        plt.plot(test_input_x, test_input_y)
        plt.pause(0.00000001)
        plt.cla()
    '''
    

    net.set_test(True)
    # テストデータの出力のaccuracyを学習ステップごとに行ってみる
    outputs = net(Variable(torch.from_numpy(X_test).float()))
    _, predicted = torch.max(outputs.data, 1)
    #print(predicted)
    y_predicted = predicted.numpy()
    y_true = np.argmax(y_test, axis=1)
    accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    print('accuracy: {0}%'.format(accuracy))


plt.ioff()
#plt.show()


plt.plot( loss_list )
plt.show()

# utility function to predict for an unknown data


def predict(X):
    X = Variable(torch.from_numpy(np.array(X)).float())
    outputs = net(X)
    return np.argmax(outputs.data.numpy())
