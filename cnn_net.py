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
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):

    def __init__(self, Y, X, settings):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.Y = Y
        self.train_X = X
        self.settings = settings
        # バンド幅も推定する
        self.h = nn.Parameter(torch.tensor(1.5, requires_grad=True))

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# データの用意
'''
iris = datasets.load_digits()
y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
y[np.arange(len(iris.target)), iris.target] = 1
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, y, test_size=0.9)
print(len(X_train))

x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)

# leave_one_outのために定数として一応用意しておく
x_static = Variable(torch.from_numpy(X_train).float(), requires_grad=False)


y = Variable(torch.from_numpy(y_train).float())
'''

# args = parser()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])


trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=2)


testset = torchvision.datasets.MNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=2)


classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

# leave one outの計算のため、事前に入力と出力のパラメータをセットしておく
# net = Net(y, x_static, {"activation": "leave_one_out"})

net = Net(0, 0, 0)
optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.MSELoss()

'''
test_input_x = np.linspace(-20, 20, 200)
test_input_x_list = []
for p in test_input_x:
    test_input_x_list.append([p, p, p, p, p, p, p, p, p, p])

test_input_x_torch = torch.from_numpy(np.array(test_input_x_list)).float()

plt.ion()
'''


for epoch in range(10000):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[{:d}, {:5d}] loss: {:.3f}'
                  .format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

'''
    if(i % 1 == 0):
        test_input_y_torch = net.leave_one_out(test_input_x_torch)
        test_input_y = test_input_y_torch.to('cpu').detach().numpy().copy()

        plt.plot(test_input_x, test_input_y)
        plt.pause(0.00000001)
        plt.cla()
    '''


plt.ioff()
plt.show()

# テストデータの出力のaccuracyを学習ステップごとに行ってみる
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
