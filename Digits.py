import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from matplotlib import style

print(torch.cuda.is_available())

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 10)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)



MODEL_NAME = f"model-{int(time.time())}"  # gives a dynamic model name, to just help with things getting messy over time.

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


net = Net().to(device)
net.load_state_dict(torch.load('./digit_model.pt'))
print(net)
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

PX=[]

with torch.no_grad():
    for i in range(10):
        img = cv2.imread("./n"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE)
        pltimg=np.array(img)
        imgplot = plt.imshow(pltimg)
        img = cv2.resize(img, (50, 50))
        plt.show()
        PX.append(img)
        PX[i]=torch.Tensor(PX[i])
        PX[i]= PX[i]/255.0

        net_out = net(PX[i].view(-1, 1, 50, 50).to(device))
        print(net_out)
        predicted_class = torch.argmax(net_out)
        print(predicted_class)
        if int(predicted_class)==0:
            print("Cero")
        if int(predicted_class)==1:
            print("Uno")
        if int(predicted_class)==2:
            print("Dos")
        if int(predicted_class)==3:
            print("Tres")
        if int(predicted_class)==4:
            print("Cuatro")
        if int(predicted_class)==5:
            print("Cinco")
        if int(predicted_class)==6:
            print("Seis")
        if int(predicted_class)==7:
            print("Siete")
        if int(predicted_class)==8:
            print("Ocho")
        if int(predicted_class)==9:
            print("Nueve")



