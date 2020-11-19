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

def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(F.softplus(x))



AVE = 3
LR = 0.000001
STEP = 500


'''
DATA_TYPE = "label"
DATA_OUTPUT_LENGTH = 3
DATA_MID_LENGTH = 40
DATA_INPUT_LENGTH = 13
iris = datasets.load_wine()
'''

'''
DATA_TYPE = "label"
DATA_OUTPUT_LENGTH = 3
DATA_MID_LENGTH = 3
DATA_INPUT_LENGTH = 4
iris = datasets.load_iris()
'''

'''
DATA_TYPE = "label"
DATA_OUTPUT_LENGTH = 10
DATA_MID_LENGTH = 100
DATA_INPUT_LENGTH = 64
iris = datasets.load_digits()
'''


DATA_TYPE = "reg"
DATA_OUTPUT_LENGTH = 1
DATA_MID_LENGTH = 20
DATA_INPUT_LENGTH = 13
iris = datasets.load_boston()


loss_list = []
loss_sigmoid_list = []
loss_relu_list = []
acc_list = []



def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

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
        if(abs(x - res) < abs(nearest - res)):
            nearest = k
        if(res <= x + 0.5 and res >= x - 0.5):
            return k
        elif (res is None):
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
        self.calc = False
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
            input_data = torch.from_numpy(
                np.array(create_list_data(x))).float()
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
            fc2_matrix_num_temp = self.fc2.weight.to(
                'cpu').detach().numpy().copy()
            fc2_matrix = np.linalg.inv(fc2_matrix_num_temp)
            T = np.dot(fc2_matrix, M_torch)

            result.append(T.tolist())

        return torch.from_numpy(np.array(result)).clone().float()

    def create_middle_layer_y(self):

        self.M = self._create_middle_layer_y(self.Y)
        self.calc_M = self._create_middle_layer_y(self.calc_Y)

    def set_test(self, test):
        self.test = test

    def set_middle(self, middle):
        self.middle = middle

    def set_calc(self, calc):
        self.calc = calc

    def forward(self, x):

        xw = F.relu(self.fc1(x))
        xw = self.fc2(xw)

        # reluかleave_one_out切り分け
        if self.settings["activation"] == "leave_one_out":
            if(not self.test):
                y = self.leave_one_out(xw)
            if(self.test):
                y = self.leave_one_out_output(xw)
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
    y = np.zeros((len(iris.target), 1), dtype=int)
    for i, x in enumerate(iris.target):
        y[i] = [x]
    print(y)

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, y, test_size=0.2)
print(len(X_train))
print(len(X_test))

#0.2

_, X_calc, _, y_calc = train_test_split(
    iris.data, y, test_size=0.40)

#0.04
print(len(X_calc))

x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)

# leave_one_outのために定数として一応用意しておく
x_static = Variable(torch.from_numpy(X_train).float(), requires_grad=False)
x_static_calc = Variable(torch.from_numpy(X_calc).float(), requires_grad=False)


y = Variable(torch.from_numpy(y_train).float())
y_calc = Variable(torch.from_numpy(y_calc).float())

# leave one outの計算のため、事前に入力と出力のパラメータをセットしておく
net = Net(y, y_calc, x_static, x_static_calc, {"activation": "leave_one_out"})
net_sigmoid = Net(y, y_calc, x_static, x_static_calc, {"activation": "sigmoid"})
net_relu = Net(y, y_calc, x_static, x_static_calc, {"activation": "relu"})
net_linear = Net(y, y_calc, x_static, x_static_calc, {"activation": "linear"})
net_mish = Net(y, y_calc, x_static, x_static_calc, {"activation": "mish"})
net_swish = Net(y, y_calc, x_static, x_static_calc, {"activation": "swish"})

optimizer = optim.SGD(net.parameters(), lr=LR)
optimizer_sigmoid = optim.SGD(net_sigmoid.parameters(), lr=LR)
optimizer_relu = optim.SGD(net_relu.parameters(), lr=LR)
criterion = nn.MSELoss()


test_input_x = np.linspace(-50, 50, 200)
test_input_x_list = []
for p in test_input_x:
    test_input_x_list.append(create_list_data(p))

test_input_x_torch = torch.from_numpy(np.array(test_input_x_list)).float()

plt.ion()







def calc_with_net(neural_network, name)



    acc_list=[]
    for i in range(0,AVE):

        for j in range(STEP):
            optimizer_relu.zero_grad()
            output = net_mish(x)
            loss_relu = criterion(output, y)
            loss_relu.backward()
            optimizer_relu.step()
            #print(loss_relu)
            loss_relu_list.append(loss_relu)




        neural_network.set_test(True)
        outputs = neural_network(Variable(torch.from_numpy(X_test).float()))
        if DATA_TYPE=="label":
            _, predicted = torch.max(outputs.data, 1)
        if DATA_TYPE=="reg":
            predicted = outputs.data
        y_predicted = predicted.numpy()


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
            print('accumulate: {0}%'.format(accuracy))

        acc_list.append(accuracy)
        if (i != AVE-1 ): 
            neural_network = Net(y, y_calc, x_static, x_static_calc, {"activation": "relu"})
            optimizer_relu = optim.SGD(neural_network.parameters(), lr=LR)
            loss_relu_list = []



    print(name)
    print("average")
    ave = sum(acc_list) / len(acc_list)
    print(ave)

    
    return ave, acc_list, loss_relu_list















# leave one out perceptron

'''
for i in range(0,AVE):

    for j in range(STEP):
        
        net.set_test(True)
        optimizer.zero_grad()
        output = net(x)
        #print(output)
        loss = criterion(output, y)
        print(loss)
        loss.backward()
        optimizer.step()
        loss_list.append(loss)
        print(j)



        if(j % 1 == 0):
            test_input_y_torch = net.leave_one_out_output(test_input_x_torch)
            test_input_y = test_input_y_torch.to('cpu').detach().numpy().copy()

            plt.plot(test_input_x, test_input_y)
            plt.pause(0.00000001)
            plt.cla()




    
    net.set_test(True)
    outputs = net(Variable(torch.from_numpy(X_test).float()))
    if DATA_TYPE=="label":
        _, predicted = torch.max(outputs.data, 1)
    if DATA_TYPE=="reg":
        predicted = outputs.data
    y_predicted = predicted.numpy()

    if DATA_TYPE=="label":
        y_true = np.argmax(y_test, axis=1)
        accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    if DATA_TYPE=="reg":
        y_true = y_test
        accuracy = (int)(np.sum(y_predicted - y_true) / len(y_predicted))



    if DATA_TYPE=="label":
        print('accuracy: {0}%'.format(accuracy))
    if DATA_TYPE=="reg":
        print('accumulate: {0}%'.format(accuracy))

    acc_list.append(accuracy)

    
    if (i != AVE-1 ): 
        net = Net(y, y_calc, x_static, x_static_calc, {"activation": "leave_one_out"})
        optimizer = optim.SGD(net.parameters(), lr=LR)
        print("reset")
        loss_list = []

print("leave one out")
print("average")
ave = sum(acc_list) / len(acc_list)
print(ave)


'''

'''
ave, acc_list, loss_relu_list = calc_with_net(net_sigmoid, "sigmoid")
ave, acc_list, loss_relu_list = calc_with_net(net_sigmoid, "sigmoid")
ave, acc_list, loss_relu_list = calc_with_net(net_sigmoid, "sigmoid")
ave, acc_list, loss_relu_list = calc_with_net(net_sigmoid, "sigmoid")
ave, acc_list, loss_relu_list = calc_with_net(net_sigmoid, "sigmoid")
'''



acc_list=[]
for i in range(0,AVE):

    for j in range(STEP):
        optimizer_sigmoid.zero_grad()
        output = net_sigmoid(x)
        loss_sigmoid = criterion(output, y)
        loss_sigmoid.backward()
        optimizer_sigmoid.step()
        #print(loss_sigmoid)

        #print("sigmiod")
        #print(i)


        loss_sigmoid_list.append(loss_sigmoid)


    net_sigmoid.set_test(True)
    outputs = net_sigmoid(Variable(torch.from_numpy(X_test).float()))
    if DATA_TYPE=="label":
        _, predicted = torch.max(outputs.data, 1)
    if DATA_TYPE=="reg":
        predicted = outputs.data
    y_predicted = predicted.numpy()

    if DATA_TYPE=="label":
        y_true = np.argmax(y_test, axis=1)
        accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    if DATA_TYPE=="reg":
        y_true = y_test
        accuracy = (int)(np.sum(y_predicted - y_true) / len(y_predicted))



    if DATA_TYPE=="label":
        print('accuracy: {0}%'.format(accuracy))
    if DATA_TYPE=="reg":
        print('accumulate: {0}%'.format(accuracy))

    acc_list.append(accuracy)

    if (i != AVE-1 ): 
        net_sigmoid = Net(y, y_calc, x_static, x_static_calc, {"activation": "sigmoid"})
        optimizer_sigmoid = optim.SGD(net_sigmoid.parameters(), lr=LR)
        loss_sigmoid_list = []



print("sigmoid")
print("average")
ave = sum(acc_list) / len(acc_list)
print(ave)












acc_list=[]
for i in range(0,AVE):

    for j in range(STEP):
        optimizer_relu.zero_grad()
        output = net_relu(x)
        loss_relu = criterion(output, y)
        loss_relu.backward()
        optimizer_relu.step()
        #print(loss_relu)
        loss_relu_list.append(loss_relu)




    net_relu.set_test(True)
    outputs = net_relu(Variable(torch.from_numpy(X_test).float()))
    if DATA_TYPE=="label":
        _, predicted = torch.max(outputs.data, 1)
    if DATA_TYPE=="reg":
        predicted = outputs.data
    y_predicted = predicted.numpy()


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
        print('accumulate: {0}%'.format(accuracy))

    acc_list.append(accuracy)
    if (i != AVE-1 ): 
        net_relu = Net(y, y_calc, x_static, x_static_calc, {"activation": "relu"})
        optimizer_relu = optim.SGD(net_relu.parameters(), lr=LR)
        loss_relu_list = []


print("relu")
print("average")
ave = sum(acc_list) / len(acc_list)
print(ave)














acc_list=[]
for i in range(0,AVE):

    for j in range(STEP):
        optimizer_relu.zero_grad()
        output = net_linear(x)
        loss_relu = criterion(output, y)
        loss_relu.backward()
        optimizer_relu.step()
        #print(loss_relu)
        loss_relu_list.append(loss_relu)




    net_linear.set_test(True)
    outputs = net_linear(Variable(torch.from_numpy(X_test).float()))
    if DATA_TYPE=="label":
        _, predicted = torch.max(outputs.data, 1)
    if DATA_TYPE=="reg":
        predicted = outputs.data
    y_predicted = predicted.numpy()


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
        print('accumulate: {0}%'.format(accuracy))

    acc_list.append(accuracy)
    if (i != AVE-1 ): 
        net_linear = Net(y, y_calc, x_static, x_static_calc, {"activation": "relu"})
        optimizer_relu = optim.SGD(net_linear.parameters(), lr=LR)
        loss_relu_list = []


print("linear")
print("average")
ave = sum(acc_list) / len(acc_list)
print(ave)













acc_list=[]
for i in range(0,AVE):

    for j in range(STEP):
        optimizer_relu.zero_grad()
        output = net_swish(x)
        loss_relu = criterion(output, y)
        loss_relu.backward()
        optimizer_relu.step()
        #print(loss_relu)
        loss_relu_list.append(loss_relu)




    net_swish.set_test(True)
    outputs = net_swish(Variable(torch.from_numpy(X_test).float()))
    if DATA_TYPE=="label":
        _, predicted = torch.max(outputs.data, 1)
    if DATA_TYPE=="reg":
        predicted = outputs.data
    y_predicted = predicted.numpy()


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
        print('accumulate: {0}%'.format(accuracy))

    acc_list.append(accuracy)
    if (i != AVE-1 ): 
        net_swish = Net(y, y_calc, x_static, x_static_calc, {"activation": "relu"})
        optimizer_relu = optim.SGD(net_swish.parameters(), lr=LR)
        loss_relu_list = []


print("swish")
print("average")
ave = sum(acc_list) / len(acc_list)
print(ave)



















acc_list=[]
for i in range(0,AVE):

    for j in range(STEP):
        optimizer_relu.zero_grad()
        output = net_mish(x)
        loss_relu = criterion(output, y)
        loss_relu.backward()
        optimizer_relu.step()
        #print(loss_relu)
        loss_relu_list.append(loss_relu)




    net_mish.set_test(True)
    outputs = net_mish(Variable(torch.from_numpy(X_test).float()))
    if DATA_TYPE=="label":
        _, predicted = torch.max(outputs.data, 1)
    if DATA_TYPE=="reg":
        predicted = outputs.data
    y_predicted = predicted.numpy()


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
        print('accumulate: {0}%'.format(accuracy))

    acc_list.append(accuracy)
    if (i != AVE-1 ): 
        net_mish = Net(y, y_calc, x_static, x_static_calc, {"activation": "relu"})
        optimizer_relu = optim.SGD(net_mish.parameters(), lr=LR)
        loss_relu_list = []


print("mish")
print("average")
ave = sum(acc_list) / len(acc_list)
print(ave)




















'''
net.create_middle_layer_y()


net.set_middle(True)

for g in optimizer.param_groups:
    g['lr'] = 0.005

for i in range(10000):
    net.set_calc(False)
    net.set_middle(True)
    net.set_test(True)

    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, net.M)
    # print(net.M)
    print(loss)
    loss.backward()
    optimizer.step()

    #net.set_test(True)
    #pip install -U scikit-learnnet.set_middle(False)

    net.set_calc(True)

    outputs = net(Variable(torch.from_numpy(X_train).float()))
    _, predicted = torch.max(outputs.data, 1)
    y_predicted = predicted.numpy()
    y_true = np.argmax(y_train, axis=1)
    accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    print('accuracy: {0}%'.format(accuracy))

    print(i)

    if(i % 1 == 0):
        test_input_y_torch = net.middle_layer_leave_one_out_output(
            test_input_x_torch)
        test_input_y = test_input_y_torch.to('cpu').detach().numpy().copy()
        # print(test_input_y)

        plt.plot(test_input_x, test_input_y)
        plt.pause(0.0000001)
        plt.cla()

    loss_list.append(loss)
# テストデータの出力のaccuracyを学習ステップごとに行ってみる

plt.ioff()
plt.show()

net.set_test(True)
outputs = net(Variable(torch.from_numpy(X_test).float()))
_, predicted = torch.max(outputs.data, 1)
y_predicted = predicted.numpy()
y_true = np.argmax(y_test, axis=1)
accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
print('accuracy: {0}%'.format(accuracy))

outputs = net_sigmoid(Variable(torch.from_numpy(X_test).float()))
_, predicted = torch.max(outputs.data, 1)
y_predicted = predicted.numpy()
y_true = np.argmax(y_test, axis=1)
accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
print('accuracy: {0}%'.format(accuracy))

outputs = net_relu(Variable(torch.from_numpy(X_test).float()))
_, predicted = torch.max(outputs.data, 1)
y_predicted = predicted.numpy()
y_true = np.argmax(y_test, axis=1)
accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
print('accuracy: {0}%'.format(accuracy))

'''



plt.plot( loss_list )
plt.plot( loss_sigmoid_list )
plt.plot( loss_relu_list )



plt.show()


#accuracy: 88%
#accuracy: 36%
#accuracy: 16%