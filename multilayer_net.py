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

# pytorchのガウス関数

#DATA_OUTPUT_LENGTH = 3
#DATA_MID_LENGTH = 13
#DATA_INPUT_LENGTH = 13



DATA_OUTPUT_LENGTH = 3
DATA_MID_LENGTH = 3
DATA_INPUT_LENGTH = 4

#DATA_OUTPUT_LENGTH = 10
#DATA_MID_LENGTH = 10
#DATA_INPUT_LENGTH = 64

loss_list = []
acc_list = []


def gauss(x, a=1, mu=0, sigma=1):
    return a * torch.exp(-(x - mu)**2 / (2*sigma**2))

def create_list_data(p):
    l = []
    for i in range(0, DATA_OUTPUT_LENGTH):
        l.append(p)
    return l

def create_list_data_index(p, index):
    l = []
    for i in range(0, DATA_OUTPUT_LENGTH):
        if(i == index):
            l.append(p)
        else:
            l.append(0)
    return l


def predict(X):
    X = Variable(torch.from_numpy(np.array(X)).float())
    outputs = net(X)
    return np.argmax(outputs.data.numpy())



def calc_inverse(x, f):
    k = -20.0
    nearest = 0.0
    while 1:

        res = f(k)
        if( abs(x - res) < abs(nearest - res) ):
            nearest = k
        if( res <= x + 0.5 and  res >= x - 0.5 ):
            return k
        elif ( res is None ):
            k += 1
        else:
            k += 1
        if(k > 20):
            return nearest

def leave_one_out_output_inverse(x, f):
    pass

    

class Net(nn.Module):

    def __init__(self, Y, calc_Y, X, calc_X, settings):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(DATA_INPUT_LENGTH, DATA_MID_LENGTH, bias=False)
        self.fc2 = nn.Linear(DATA_MID_LENGTH, DATA_OUTPUT_LENGTH, bias=False)
        # leave_ont_outのために事前に入力と出力をセットしておく
        self.Y = Y
        self.calc_Y = calc_Y
        self.M = None
        self.calc_M = None
        self.train_X = X
        self.calc_X = calc_X
        self.settings = settings
        self.test = False
        self.middle = False
        # バンド幅も推定する
        self.h = nn.Parameter(torch.tensor(1.5, requires_grad=True))
        self.h_middle = torch.tensor(1.0)

        self.last_layer_result = []
        self.sigmoid = nn.Sigmoid()

    # leave_one_out推定量の計算

    def leave_one_out(self, Zw):
        numerator = 0
        denominator = 0
        result = []
        # print("h")
        # print(self.h)
        for j, x_j in enumerate(self.train_X):

            Xw = self.fc2(F.relu(self.fc1(x_j)))
            tmp = gauss((Xw - Zw) / self.h)

            tmp[j] = 0
            denominator += tmp
            numerator += tmp * self.Y[j]

        g = numerator/denominator
        return g



    def leave_one_out_output(self, Zw):
        numerator = 0
        denominator = 0
        result = []
        for j, x_j in enumerate(self.calc_X):

            Xw = self.fc2(F.relu(self.fc1(x_j)))
            tmp = gauss((Xw - Zw) / self.h)
            denominator += tmp
            numerator += tmp * self.calc_Y[j]

        g = numerator/denominator
        return g

    def middle_layer_leave_one_out_output(self, Zw):
        numerator = 0
        denominator = 0
        result = []
        for j, x_j in enumerate(self.calc_X):

            Xw = self.fc1(x_j)
            tmp = gauss((Xw - Zw) / self.h_middle)
            denominator += tmp
            numerator += tmp * self.calc_M[j]

        g = numerator/denominator
        return g



    def get_loo(self, index):


        def _get_loo(x):
            input_data = torch.from_numpy(np.array(create_list_data(x))).float()
            output = self.leave_one_out_output(input_data)
            result = output[index]
            
            return result 

        return _get_loo

    def _create_middle_layer_y(self, data):

        result = []
        print(len(data))
        for d in data:
            M = []
            for i in range(0, DATA_OUTPUT_LENGTH):
                loo = self.get_loo(i)
                M.append(calc_inverse(d[i], loo))
            

            M_torch = np.array(M)
            fc2_matrix_num_temp = self.fc2.weight.to('cpu').detach().numpy().copy()
            fc2_matrix = np.linalg.inv( fc2_matrix_num_temp )
            T = np.dot(fc2_matrix , M_torch)

            result.append(T.tolist())

        return torch.from_numpy(np.array(result)).clone().float() 

    def create_middle_layer_y(self):

        self.M = self._create_middle_layer_y(self.Y)
        self.calc_M = self._create_middle_layer_y(self.calc_Y)

        


    def set_test(self, test):
        self.test = test

    def set_middle(self, middle):
        self.middle = middle


    def forward(self, x):
        if (self.middle):
            xw = self.fc1(x)
            print(self.fc1.weight)
            xw1 = self.middle_layer_leave_one_out_output(xw)
            return xw1
            '''
            #print(xw1)
            weight = torch.from_numpy(self.fc2.weight.to('cpu').detach().numpy().copy()).float()
            #print(weight)
            res = []
            for xw_child in xw1:
                xw_child_mut =  torch.mv( weight , xw_child)
                res.append(xw_child_mut.tolist())
            xw = torch.from_numpy(np.array(res)).float()
            xw = self.fc2(xw)
            #print(xw)
            #print(self.fc2(xw1))
            print("bbbbbb")
            '''
        else:
            xw = F.relu(self.fc1(x))
            xw = self.fc2(xw)

        # reluかleave_one_out切り分け
        if self.settings["activation"] == "leave_one_out":
            if(not self.test):
                y = self.leave_one_out(xw)
            if(self.test):
                y = self.leave_one_out_output(xw)
        else:
            y = self.sigmoid(xw)

        return y


# データの用意

#iris = datasets.load_digits()
iris = datasets.load_iris()
#iris = datasets.load_wine()
y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
y[np.arange(len(iris.target)), iris.target] = 1
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, y, test_size=0.9)


X_calc, _, y_calc, _ = train_test_split(
    iris.data, y, test_size=0.9)

print(len(X_calc))

x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)

# leave_one_outのために定数として一応用意しておく
x_static = Variable(torch.from_numpy(X_train).float(), requires_grad=False)
x_static_calc = Variable(torch.from_numpy(X_calc).float(), requires_grad=False)


y = Variable(torch.from_numpy(y_train).float())
y_calc = Variable(torch.from_numpy(y_calc).float())

# leave one outの計算のため、事前に入力と出力のパラメータをセットしておく
net = Net(y, y_calc, x_static, x_static_calc, {"activation": "leave_one_out"})
optimizer = optim.SGD(net.parameters(), lr=1.05)
criterion = nn.MSELoss()


test_input_x = np.linspace(-20, 20, 200)
test_input_x_list = []
for p in test_input_x:
    test_input_x_list.append(create_list_data(p))

test_input_x_torch = torch.from_numpy(np.array(test_input_x_list)).float()



plt.ion()

for i in range(100):
    net.set_test(True)
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    
    loo = net.get_loo(0)

    
    net.set_test(True)
    outputs = net(Variable(torch.from_numpy(X_test).float()))
    _, predicted = torch.max(outputs.data, 1)
    y_predicted = predicted.numpy()
    y_true = np.argmax(y_test, axis=1)
    accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    print('accuracy: {0}%'.format(accuracy))
    

    print(i)
    
    if(i % 1 == 0):
        test_input_y_torch = net.leave_one_out_output(test_input_x_torch)
        test_input_y = test_input_y_torch.to('cpu').detach().numpy().copy()
        #print(test_input_y)

        plt.plot(test_input_x, test_input_y)
        plt.pause(0.00000001)
        plt.cla()
    
    
    
    loss_list.append(loss)

    # テストデータの出力のaccuracyを学習ステップごとに行ってみる


net.create_middle_layer_y()


net.set_middle(True)

for g in optimizer.param_groups:
    g['lr'] = 0.01

for i in range(1000):
    net.set_middle(True)
    net.set_test(True)
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, net.M)
    #print(net.M)
    print(loss)
    loss.backward()
    optimizer.step()

    
    '''
    net.set_test(True)
    net.set_middle(False)

    outputs = net(Variable(torch.from_numpy(X_test).float()))
    _, predicted = torch.max(outputs.data, 1)
    y_predicted = predicted.numpy()
    y_true = np.argmax(y_test, axis=1)
    accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    print('accuracy: {0}%'.format(accuracy))
    
    '''

    print(i)
    
    if(i % 1 == 0):
        test_input_y_torch = net.middle_layer_leave_one_out_output(test_input_x_torch)
        test_input_y = test_input_y_torch.to('cpu').detach().numpy().copy()
        #print(test_input_y)

        plt.plot(test_input_x, test_input_y)
        plt.pause(0.0000001)
        plt.cla()
    
    
    
    loss_list.append(loss)

    # テストデータの出力のaccuracyを学習ステップごとに行ってみる

plt.ioff()
plt.show()

'''
net.set_test(True)
outputs = net(Variable(torch.from_numpy(X_test).float()))
_, predicted = torch.max(outputs.data, 1)
y_predicted = predicted.numpy()
y_true = np.argmax(y_test, axis=1)
accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
print('accuracy: {0}%'.format(accuracy))


plt.plot( loss_list )
plt.show()
'''
