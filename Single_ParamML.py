import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import random as rand
import sys, os, time
import seaborn as sns

#Defines Architecture - More work -Normalize Data before CNN Specifcally th images
#output size  = (input size + 2 * padding - kernel)/stride + 1
#output size MUST be integer
#Use CNNs to reduce images to at least 4x4,2x2, or 1x1
#Tiger Folder location /projects/QUIJOTE/jsalmon
#weighdecay use differnt values to get loss to .0001
#save data and make plots with loss
#Save model and Respecitive losses
#modify code to use 2 gpus optional
#take log of all fields excluding stellar mass(log(x+1))
class Net(nn.Module):
    def __init__(self):
        n = 1
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n,out_channels=4*n,kernel_size= 4,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4*n, out_channels=8*n,kernel_size=5 ,stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8*n, out_channels=16*n,kernel_size=4 ,stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(in_channels=16*n, out_channels=32*n,kernel_size=5 ,stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(in_channels=32*n, out_channels=64*n,kernel_size=5 ,stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(in_channels=64*n, out_channels=128*n,kernel_size=5 ,stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(in_channels=128*n, out_channels=256*n,kernel_size=5 ,stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*n, 128*n)
        self.fc2 = nn.Linear(128 * n, 1)

    #Max pool and stride function similarly(Data lost with Max_pool)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Trains Net(10 Epochs, also times)
# increase epoch number start with 100
def train(net,numEpochs):
    start = time.time()
    for epoch in range(numEpochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Loss From Training:"+str(loss))
    end = time.time()

    print('Done Training')
    print('%0.2f minutes' % ((end - start) / 60))


# This class creates the dataset
class make_dataset():

    def __init__(self, mode, seed, f_images, f_params, minimum=None, maximum=None,
                 verbose=False):

        # read the data
        i = int(input('What parameter would you like to train for?'))
        print('Creating %s dataset' % mode)
        data = np.load(f_images)  # [number of maps, height, width]
        params_pre = np.loadtxt(f_params)
        params = params_pre[:,i]

        # normalize maps
        if(f_images == 'Mstar'): data = np.log10(data+1)
        else: data = np.log10(data)
        if minimum is None:  minimum = np.min(data)
        if maximum is None:  maximum = np.max(data)
        if verbose:  print('%.3f < T(all|input) < %.3f' % (np.min(data), np.max(data)))
        data = 2 * (data - minimum) / (maximum - minimum) - 1.0
        if verbose:  print('%.3f < T(all|norm) < %.3f' % (np.min(data), np.max(data)))

        # normalize params
        minimum_pre = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum_pre = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        minimum = minimum_pre[i]
        maximum = maximum_pre[i]
        params = (params - minimum) / (maximum - minimum)

        # get the size and offset depending on the type of dataset
        unique_maps = data.shape[0]
        if mode == 'train':
            size, offset = int(unique_maps * 0.70), int(unique_maps * 0.00)
        elif mode == 'valid':
            size, offset = int(unique_maps * 0.15), int(unique_maps * 0.70)
        elif mode == 'test':
            size, offset = int(unique_maps * 0.15), int(unique_maps * 0.85)
        elif mode == 'all':
            size, offset = int(unique_maps * 1.00), int(unique_maps * 0.00)
        else:
            raise Exception('Wrong name!')

        # randomly shuffle the maps. Instead of 0 1 2 3...999 have a
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(unique_maps)  # only shuffle realizations, not rotations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset + size]  # select indexes of mode

        # keep only the data with the corresponding indexes
        data = data[indexes]
        params = params[indexes]

        # define the matrix hosting all data with all rotations/flipping
        # together with the array containing the numbers of each map
        data_all = np.zeros((size * 8, data.shape[1], data.shape[2]), dtype=np.float32)
        params_all = np.zeros((size * 8, len(params)), dtype=np.float32)

        # do a loop over all rotations (each is 90 deg)
        total_maps = 0
        for rot in [0, 1, 2, 3]:
            data_rot = np.rot90(data, k=rot, axes=(1, 2))

            data_all[total_maps:total_maps + size, :, :] = data_rot
            params_all[total_maps:total_maps + size] = params
            total_maps += size

            data_all[total_maps:total_maps + size, :, :] = np.flip(data_rot, axis=1)
            params_all[total_maps:total_maps + size] = params
            total_maps += size

        if verbose:
            print('This set contains %d maps' % total_maps)
            print('%.3f < T (this set) < %.3f\n' % (np.min(data), np.max(data)))

        self.size = data_all.shape[0]
        self.x = torch.unsqueeze(torch.tensor(data_all, dtype=torch.float32), 1)
        self.y = torch.tensor(params_all, dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# This routine creates the dataset
def create_dataset(mode, seed, f_images, f_params, batch_size, minimum, maximum,
                   verbose=False):
    data_set = make_dataset(mode, seed, f_images, f_params, minimum, maximum, verbose)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return data_loader

#Loads Data as np array( Data is not Normalized)
#Take log of all data
#Compute Mean of all images (np.mean)
#Compute STD of all images (np.std)
#Define New Field(T_Maps_Norm)
#T_Maps_Norm = (T_maps-mean)/std
#Take log10 of data and normalize sub data set

#Creates 1st Heatmap
i = int(input('What parameter would you like to train for?'))
f_img = ""
invalid =True
while(invalid == True):
    img = input("Enter Type of Images to train and test on.(T,ne,P,Z,Vgas,Mstar,HI,Mcdm,Mgas)")
    if(img =='T'):
        f_img = 'data/Images_T_IllustrisTNG.npy'
        invalid = False
    elif (img == 'ne'):
        f_img = 'data/Images_ne_IllustrisTNG.npy'
        invalid = False
    elif (img == 'P'):
        f_img = 'data/Images_P_IllustrisTNG.npy'
        invalid = False
    elif (img == 'Z'):
        f_img = 'data/Images_Z_IllustrisTNG.npy'
        invalid = False
    elif (img == 'Vgas'):
        f_img = 'data/Images_Vgas_IllustrisTNG.npy'
        invalid = False
    elif (img == 'Mstar'):
        f_img = 'data/Images_Mstar_IllustrisTNG.npy'
        invalid = False
    elif (img == 'Mcdm'):
        f_img = 'data/Images_Mcdm_IllustrisTNG.npy'
        invalid = False
    elif (img == 'HI'):
        f_img = 'data/Images_HI_IllustrisTNG.npy'
        invalid = False
    elif (img == 'Mgas'):
        f_img = 'data/Images_Mgas_IllustrisTNG.npy'
        invalid = False
    else:
        print('Entry not valid. Please enter valid field')
#Creating Datasets
trainloader = create_dataset(mode = 'train', seed = 1, f_images = f_img,
                             f_params = 'data/Cosmo_astro_params_IllustrisTNG.txt',minimum = None , maximum = None
                             ,batch_size = int(input("Enter Training batch_size : ")))
validloader = create_dataset(mode = 'valid', seed=1, f_images = f_img,
                             f_params = 'data/Cosmo_astro_params_IllustrisTNG.txt',
                             batch_size = int(input("Enter Validation batch_size : ")),
                             minimum=None, maximum=None, verbose=True)
testloader = create_dataset(mode = 'test', seed = 1, f_images = f_img,
                             f_params = 'data/Cosmo_astro_params_IllustrisTNG.txt',minimum = None , maximum = None
                             ,batch_size = int(input("Enter Test batch_size : ")))


print("Data Loaded Successfully")

# Sets parameters and GPU if Present
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
#MSE for Regression CEL for classification
criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=float(input('Enter Learning Rate: ')), weight_decay= float(input('Enter Weight Decay: ')), betas=(0.5, 0.999))
EpochNum= int(input("Enter Number of Epochs: "))
name = img + '_Jsalmon-100-Single Param'
fout = 'losses/%s.txt' % name
fmodel = 'models/%s.pt' % name

# get validation loss
print('Computing initial validation loss')
net.eval()
min_valid_loss, points = 0.0, 0
for x, y in validloader:
    with torch.no_grad():
        x = x.to(device=device)
        y = y.to(device=device)
        y_NN = net(x)
        min_valid_loss += (criterion(y_NN, y).item()) * x.shape[0]
        points += x.shape[0]
min_valid_loss /= points
print('Initial valid loss = %.3e' % min_valid_loss)

# see if results for this model are available
offset = 0

# do a loop over all epochs
start = time.time()
trainLossCounter=[]
validLossCounter=[]
testLossCounter=[]
for epoch in range(offset, offset + EpochNum):

    # do training
    train_loss, points = 0.0, 0
    net.train()
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        y_NN = net(x)

        loss = criterion(y_NN, y)
        train_loss += (loss.item()) * x.shape[0]
        points += x.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= points
    trainLossCounter.append(train_loss)
    # do validation
    valid_loss, points = 0.0, 0
    net.eval()
    for x, y in validloader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_NN = net(x)
            valid_loss += (criterion(y_NN, y).item()) * x.shape[0]
            points += x.shape[0]
    valid_loss /= points
    validLossCounter.append(valid_loss)

    # do testing
    test_loss, points = 0.0, 0
    net.eval()
    Prediction_0=[]
    Prediction_1=[]
    Prediction_2=[]
    Prediction_3=[]
    Prediction_4=[]
    Prediction_5=[]
    Actual_0=[]
    Actual_1=[]
    Actual_2=[]
    Actual_3=[]
    Actual_4=[]
    Actual_5=[]
    for x, y in testloader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            Actual_0.append(y.tolist()[0])
            Actual_1.append(y.tolist()[1])
            Actual_2.append(y.tolist()[2])
            Actual_3.append(y.tolist()[3])
            Actual_4.append(y.tolist()[4])
            Actual_5.append(y.tolist()[5])
            y_NN = net(x)
            Prediction_0.append(y_NN.tolist()[0])
            Prediction_1.append(y_NN.tolist()[1])
            Prediction_2.append(y_NN.tolist()[2])
            Prediction_3.append(y_NN.tolist()[3])
            Prediction_4.append(y_NN.tolist()[4])
            Prediction_5.append(y_NN.tolist()[5])
            test_loss += (criterion(y_NN, y).item()) * x.shape[0]
            points += x.shape[0]
    test_loss /= points
    testLossCounter.append(test_loss)
np.savetxt('Training Loss Single Param '+img,trainLossCounter,delimiter=',')
np.savetxt('Validation Loss Single Param '+img,validLossCounter,delimiter=',')
np.savetxt('Test Loss Single Param '+img,testLossCounter,delimiter=',')
np.savetxt('Prediction_0 Single Param '+img,np.array(Prediction_0),delimiter=',')
np.savetxt('Prediction_1 Single Param '+img,np.array(Prediction_1),delimiter=',')
np.savetxt('Prediction_2 Single Param '+img,np.array(Prediction_2),delimiter=',')
np.savetxt('Prediction_3 Single Param '+img,np.array(Prediction_3),delimiter=',')
np.savetxt('Prediction_4 Single Param '+img,np.array(Prediction_4),delimiter=',')
np.savetxt('Prediction_5 Single Param '+img,np.array(Prediction_5),delimiter=',')
np.savetxt('Actual_0 Single Param '+img,np.array(Actual_0),delimiter=',')
np.savetxt('Actual_1 Single Param '+img,np.array(Actual_1),delimiter=',')
np.savetxt('Actual_2 Single Param '+img,np.array(Actual_2),delimiter=',')
np.savetxt('Actual_3 Single Param '+img,np.array(Actual_3),delimiter=',')
np.savetxt('Actual_4 Single Param '+img,np.array(Actual_4),delimiter=',')
np.savetxt('Actual_5 Single Param '+img,np.array(Actual_5),delimiter=',')
print("Test Loss : "+str(test_loss))
print("Validation Loss : "+str(valid_loss))
print("Training Loss : "+str(train_loss))
if valid_loss < min_valid_loss:
    torch.save(net.state_dict(), fmodel)
    min_valid_loss = valid_loss
    print('%03d %.3e %.3e %.3e (saving)' % (epoch, train_loss, valid_loss, test_loss))
else:
    print('%03d %.3e %.3e %.3e' % (epoch, train_loss, valid_loss, test_loss))
f = open(fout, 'a')
f.write('%d %.5e %.5e %.5e\n' % (epoch, train_loss, valid_loss, test_loss))
f.close()
print("Exit")