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
LR = 0.01

STEP = 1000
max_norm=0.25
use_clipping = True
show_activation_kernel = False


#classes = ['wine', 'iris', 'mnist', 'boston', 'diabetes', 'linnerud']
select_data="boston"
CALC_SIZE = 0.15

#mnist = 0.02 = 36個
#mnist = 0.02 = 36個

#classes = ['SGD', 'Adagrad', 'RMSprop', 'Adadelta', 'Adam', 'AdamW']
optim_method="SGD"

#classes = ['orthogonal_', 'sparse_', 'kaiming_normal_', 'kaiming_uniform_', 'dirac_', 'zeros_', 'ones_']
init_method = 'kaiming_normal_'

#classes = ['non', 'l1', 'l2']
reg_method = 'non'


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


if select_data=="diabetes":
    DATA_TYPE = "reg"
    DATA_OUTPUT_LENGTH = 1
    DATA_MID_LENGTH = 20
    DATA_INPUT_LENGTH = 10
    iris = datasets.load_diabetes()


if select_data=="breast_cancer":
    DATA_TYPE = "label"
    DATA_OUTPUT_LENGTH = 2
    DATA_MID_LENGTH = 20
    DATA_INPUT_LENGTH = 30
    iris = datasets.load_breast_cancer()



def create_optim(model):
    weight_decay=0

    if reg_method == 'l2':
        weight_decay = 0.01

    if optim_method == 'SGD':
        return optim.SGD(model.parameters(), lr=LR, weight_decay=weight_decay )
    if optim_method == 'Adagrad':
        return optim.Adagrad(model.parameters(), lr=LR, weight_decay=weight_decay )
    if optim_method == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=LR, weight_decay=weight_decay )
    if optim_method == 'Adadelta':
        return optim.Adadelta(model.parameters(), lr=LR, weight_decay=weight_decay )
    if optim_method == 'Adam':
        return optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay )
    if optim_method == 'AdamW':
        return optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay )


def init_weights(m):
    
    if type(m) == nn.Linear:
        if init_method == 'orthogonal_':
            torch.nn.init.orthogonal_(m.weight)
        if init_method == 'sparse_':
            torch.nn.init.sparse_(m.weight, sparsity=1)
        if init_method == 'kaiming_normal_':
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if init_method == 'xavier_uniform':
            torch.nn.init.xavier_uniform(m.weight)
        if init_method == 'kaiming_uniform_':
            torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if init_method == 'dirac_':
            torch.nn.init.dirac_(m.weight)
        if init_method == 'zeros_':
            torch.nn.init.zeros_(m.weight)
        if init_method == 'ones_':
            torch.nn.init.ones_(m.weight)


        #m.bias.data.fill_(0.01)


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
    error_count=0
    for i in range(0,AVE):

        for j in range(STEP):

            data_len = len(x)

            net_optimizer.zero_grad()

            output = neural_network(x)
            loss = criterion(output, y)
            loss.backward()
            
            if use_clipping:
                nn.utils.clip_grad_norm_(neural_network.parameters(), max_norm)
            
            test_output = neural_network(torch.from_numpy(X_test).float())
            test_loss = criterion(test_output, torch.from_numpy(y_test).float())
            loss_list.append(test_loss.item())
            
            net_optimizer.step()
            #if j > STEP/100:
            #loss_list.append(loss.item())
            #print(loss)
            if(name == "kernel"):
                if j % 1 == 0:
                    print(j)
            

            
            if(j % 1 == 0 and name == "kernel" and show_activation_kernel):
                test_input_y_torch = neural_network.kernel_output(test_input_x_torch)
                test_input_y = test_input_y_torch.to('cpu').detach().numpy().copy()

                plt.plot(test_input_x, test_input_y)

                plt.xlabel('input')
                plt.ylabel('outuput') 
                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
                plt.pause(0.00000001)
                plt.cla()
            

        neural_network.set_test(False)
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
            try:
                dif = y_predicted - y_true
                accuracy = (int)(np.sum(dif * dif) / len(y_predicted))
            except:       
                error_count+=1     
                neural_network = Net(y, y_calc, x_static, x_static_calc, {"activation": name})
                net_optimizer = create_optim(neural_network)

                loss_list = []
                continue #your handling code
                    


        if DATA_TYPE=="label":
            print('accuracy: {0}%'.format(accuracy))
        if DATA_TYPE=="reg":
            print('accumulate: {0}'.format(accuracy))

        acc_list.append(accuracy)

        if (i != AVE-1 ): 
            neural_network = Net(y, y_calc, x_static, x_static_calc, {"activation": name})
            net_optimizer = create_optim(neural_network)
            loss_list = []


    try:
        ave = sum(acc_list) / len(acc_list)
    except:     
        ave = 0

            


    print("average{0}", ave)
    print("AVE:{0}, error_count:{1}", AVE, error_count)
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
        self.tanh = nn.Tanh()

    # kernel推定量の計算

    def kernel(self, Zw):
        numerator = 0
        denominator = 0
        result = []
        # print("h")
        # print(self.h)
        for j, x_j in enumerate(self.calc_X):

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
            y = self.kernel_output(xw)
        elif self.settings["activation"] == "sigmoid":
            y = self.sigmoid(xw)
        elif self.settings["activation"] == "tanh":
            y = self.tanh(xw)
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

if DATA_TYPE=="reg":
    y = np.zeros((len(iris.target), DATA_OUTPUT_LENGTH), dtype=int)
    for i, x in enumerate(iris.target):
        if DATA_OUTPUT_LENGTH == 1:
            y[i] = [x]
        else:
            y[i] = x

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, y, test_size=0.2)
    
_, X_calc, _, y_calc = train_test_split(
    iris.data, y, test_size=CALC_SIZE)

    
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
net_tanh = Net(y, y_calc, x_static, x_static_calc, {"activation": "tanh"})
net_relu = Net(y, y_calc, x_static, x_static_calc, {"activation": "relu"})
net_linear = Net(y, y_calc, x_static, x_static_calc, {"activation": "linear"})
net_mish = Net(y, y_calc, x_static, x_static_calc, {"activation": "mish"})
net_swish = Net(y, y_calc, x_static, x_static_calc, {"activation": "swish"})



optimizer_kernel = create_optim(net_kernel)
optimizer_sigmoid = create_optim(net_sigmoid)
optimizer_tanh = create_optim(net_tanh)
optimizer_relu = create_optim(net_relu)
optimizer_linear = create_optim(net_linear)
optimizer_mish = create_optim(net_mish)
optimizer_swish = create_optim(net_swish)




criterion = nn.MSELoss()


test_input_x = np.linspace(-50, 50, 200)
test_input_x_list = []
for p in test_input_x:
    test_input_x_list.append(create_list_data(p))

test_input_x_torch = torch.from_numpy(np.array(test_input_x_list)).float()












# leave one out perceptron




#plt.ion()
plt.ioff()
ave_kernel, acc_list_kernel, loss_list_kernel = train(net_kernel, optimizer_kernel, "kernel", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc,)

ave_sigmoid, acc_list_sigmoid, loss_list_sigmoid = train(net_sigmoid, optimizer_sigmoid, "sigmoid", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc, )
ave_tanh, acc_list_tanh, loss_list_tanh = train(net_tanh, optimizer_tanh, "tanh", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc, )
ave_relu, acc_list_relu, loss_list_relu = train(net_relu, optimizer_relu, "relu", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc, )
#ave_linear, acc_list_linear, loss_list_linear = train(net_linear, optimizer_linear, "linear", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc, )
ave_mish, acc_list_mish, loss_list_mish = train(net_mish, optimizer_mish, "mish", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc,)
ave_swish, acc_list_swish, loss_list_swish = train(net_swish, optimizer_swish, "swish", X_train, X_test, y_train, y_test, y, y_calc, x_static, x_static_calc,)






plt.plot( loss_list_kernel[100:], label='kernel')
#plt.plot( loss_list_linear, label='linear')
plt.plot( loss_list_sigmoid[100:], label='sigmoid')
plt.plot( loss_list_tanh[100:], label='tanh')
plt.plot( loss_list_relu[100:], label='relu')
plt.plot( loss_list_mish[100:], label='mish')
plt.plot( loss_list_swish[100:], label='swish')
plt.xlabel('step')  # x軸のラベルづけ
plt.ylabel('loss')  # y軸のラベルづけ
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.show()
