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


AVE = 3
LR = 0.0000001
STEP = 200

show_activation_kernel = False


select_data="linnerud"

if select_data=="wine":
    DATA_TYPE = "label"
    DATA_OUTPUT_LENGTH = 3
    DATA_MID_LENGTH = 40
    DATA_INPUT_LENGTH = 13
    iris = datasets.load_wine()


if select_data=="iris":
    DATA_TYPE = "label"
    DATA_OUTPUT_LENGTH = 3
    DATA_MID_LENGTH = 3
    DATA_INPUT_LENGTH = 4
    iris = datasets.load_iris()


if select_data=="mnist":
    DATA_TYPE = "label"
    DATA_OUTPUT_LENGTH = 10
    DATA_MID_LENGTH = 100
    DATA_INPUT_LENGTH = 64
    iris = datasets.load_digits()
    

if select_data=="boston":
    DATA_TYPE = "reg"
    DATA_OUTPUT_LENGTH = 1
    DATA_MID_LENGTH = 20
    DATA_INPUT_LENGTH = 13
    iris = datasets.load_boston()


if select_data=="diabetes":
    DATA_TYPE = "reg"
    DATA_OUTPUT_LENGTH = 1
    DATA_MID_LENGTH = 20
    DATA_INPUT_LENGTH = 10
    iris = datasets.load_diabetes()


if select_data=="linnerud":
    DATA_TYPE = "reg"
    DATA_OUTPUT_LENGTH = 3
    DATA_MID_LENGTH = 10
    DATA_INPUT_LENGTH = 3
    iris = datasets.load_linnerud()




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




def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(F.softplus(x))




def train(neural_network, net_optimizer, name, X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc ):
    print("-----------start--------------")
    print(name)

    loss_list = []
    acc_list=[]
    for i in range(0,AVE):
        neural_network.set_test(False)

        for j in range(STEP):
            net_optimizer.zero_grad()
            output = neural_network(x)
            loss = criterion(output, y)
            loss.backward()
            net_optimizer.step()
            #if j > STEP/100:
            loss_list.append(loss.item())
            #print(loss)
            if(name == "kernel"):
                print(j)

            
            if(j % 1 == 0 and name == "kernel" and show_activation_kernel):
                test_input_y_torch = neural_network.kernel_output(test_input_x_torch)
                test_input_y = test_input_y_torch.to('cpu').detach().numpy().copy()

                plt.plot(test_input_x, test_input_y)
                plt.pause(0.00000001)
                plt.cla()
            

        neural_network.set_test(True)
        print(X_test)
        outputs = neural_network(Variable(torch.from_numpy(X_test).float()))
        print(outputs)
        if DATA_TYPE=="label":
            _, predicted = torch.max(outputs.data, 1)
        if DATA_TYPE=="reg":
            predicted = outputs.data
        y_predicted = predicted.numpy()
        print(y_predicted)


        if DATA_TYPE=="label":
            y_true = np.argmax(y_test, axis=1)
            accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
        if DATA_TYPE=="reg":
            y_true = y_test
            #print(y_test)
            #print(y_predicted)
            accuracy = (int)(np.sum(y_predicted - y_true) / len(y_predicted))
            


        if DATA_TYPE=="label":
            print('accuracy: {0}%'.format(accuracy))
        if DATA_TYPE=="reg":
            print('accumulate: {0}'.format(accuracy))

        acc_list.append(accuracy)

        if (i != AVE-1 ): 
            neural_network = Net(y, y_calc, x_static, x_static_calc, {"activation": name})
            net_optimizer = optim.SGD(neural_network.parameters(), lr=LR)
            loss_list = []
            acc_list=[]



    ave = sum(acc_list) / len(acc_list)
    print("average{0}", ave)
    print("-----------finish--------------")
    
    return ave, acc_list, loss_list












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
        self.calc = False
        # バンド幅も推定する
        self.h = nn.Parameter(torch.tensor(1.5, requires_grad=True))
        self.h_middle = torch.tensor(1.0)

        self.last_layer_result = []
        self.sigmoid = nn.Sigmoid()

    # kernel推定量の計算

    def kernel(self, Zw):
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

    def kernel_output(self, Zw):
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


    def set_test(self, test):
        self.test = test

    def set_middle(self, middle):
        self.middle = middle

    def set_calc(self, calc):
        self.calc = calc

    def forward(self, x):

        xw = F.relu(self.fc1(x))
        xw = self.fc2(xw)

        # reluかkernel切り分け
        if self.settings["activation"] == "kernel":
            if(not self.test):
                y = self.kernel(xw)
            if(self.test):
                y = self.kernel_output(xw)
        elif self.settings["activation"] == "sigmoid":
            y = self.sigmoid(xw)
        elif self.settings["activation"] == "relu":
            y = F.relu(xw)
        elif self.settings["activation"] == "linear":
            y = xw
        elif self.settings["activation"] == "swish":
            y = swish(xw)
        elif self.settings["activation"] == "mish":
            y = mish(xw)

        return y


# データの用意


if DATA_TYPE=="label":
    y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
    y[np.arange(len(iris.target)), iris.target] = 1
    print(y)

if DATA_TYPE=="reg":
    y = np.zeros((len(iris.target), DATA_OUTPUT_LENGTH), dtype=int)
    for i, x in enumerate(iris.target):
        print(x)
        if DATA_OUTPUT_LENGTH == 1:
            y[i] = [x]
        else:
            y[i] = x

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, y, test_size=0.2)
    
_, X_calc, _, y_calc = train_test_split(
    iris.data, y, test_size=0.40)

    
print(len(X_calc))

x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)

# kernelのために定数として一応用意しておく
x_static = Variable(torch.from_numpy(X_train).float(), requires_grad=False)
x_static_calc = Variable(torch.from_numpy(X_calc).float(), requires_grad=False)

y = Variable(torch.from_numpy(y_train).float())
y_calc = Variable(torch.from_numpy(y_calc).float())



# leave one outの計算のため、事前に入力と出力のパラメータをセットしておく
net_kernel = Net(y, y_calc, x_static, x_static_calc, {"activation": "kernel"})
net_sigmoid = Net(y, y_calc, x_static, x_static_calc, {"activation": "sigmoid"})
net_relu = Net(y, y_calc, x_static, x_static_calc, {"activation": "relu"})
net_linear = Net(y, y_calc, x_static, x_static_calc, {"activation": "linear"})
net_mish = Net(y, y_calc, x_static, x_static_calc, {"activation": "mish"})
net_swish = Net(y, y_calc, x_static, x_static_calc, {"activation": "swish"})



optimizer_kernel = optim.SGD(net_kernel.parameters(), lr=LR)
optimizer_sigmoid = optim.SGD(net_sigmoid.parameters(), lr=LR)
optimizer_relu = optim.SGD(net_relu.parameters(), lr=LR)
optimizer_linear = optim.SGD(net_linear.parameters(), lr=LR)
optimizer_mish = optim.SGD(net_mish.parameters(), lr=LR)
optimizer_swish = optim.SGD(net_swish.parameters(), lr=LR)




criterion = nn.MSELoss()


test_input_x = np.linspace(-50, 50, 200)
test_input_x_list = []
for p in test_input_x:
    test_input_x_list.append(create_list_data(p))

test_input_x_torch = torch.from_numpy(np.array(test_input_x_list)).float()












# leave one out perceptron




#plt.ion()
ave_kernel, acc_list_kernel, loss_list_kernel = train(net_kernel, optimizer_kernel, "kernel", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc,)
ave_sigmoid, acc_list_sigmoid, loss_list_sigmoid = train(net_sigmoid, optimizer_sigmoid, "sigmoid", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc, )
ave_relu, acc_list_relu, loss_list_relu = train(net_relu, optimizer_relu, "relu", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc, )
ave_linear, acc_list_linear, loss_list_linear = train(net_linear, optimizer_linear, "linear", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc, )
ave_mish, acc_list_mish, loss_list_mish = train(net_mish, optimizer_mish, "mish", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc,)
ave_swish, acc_list_swish, loss_list_swish = train(net_swish, optimizer_swish, "swish", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc,)






plt.plot( loss_list_kernel, label='kernel')
plt.plot( loss_list_linear, label='linear')
plt.plot( loss_list_relu, label='relu')
plt.plot( loss_list_mish, label='mish')
plt.plot( loss_list_swish, label='swish')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.show()





