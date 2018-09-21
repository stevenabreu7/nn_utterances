import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import optparse

# Parse options
parser = optparse.OptionParser()

parser.add_option('-e', '--epochs', action="store", dest="epochs", help="number of training epochs", default=20)
parser.add_option('-r', '--rate', action="store", dest="rate", help="learning rate", default=0.1)
parser.add_option('-b', '--batchsize', action="store", dest="batch_size", help="batch size", default=500)

options, args = parser.parse_args()

# Load data
print('Loading data')

trainx = np.load('data/train.npy', encoding='bytes')
trainy = np.load('data/train_labels.npy', encoding='bytes')
testx = np.load('data/test.npy', encoding='bytes')
valx = np.load('data/dev.npy', encoding='bytes')
valy = np.load('data/dev_labels.npy', encoding='bytes')

# Preprocessing data
print('Preprocessing data (concatenating)')

trainx = np.concatenate(trainx.tolist())
trainy = np.concatenate(trainy.tolist())
testx = np.concatenate(testx.tolist())
valx = np.concatenate(valx.tolist())
valy = np.concatenate(valy.tolist())

# Turn into tensors
print('Converting into tensors')

trainx = torch.from_numpy(trainx).float()
trainy = torch.from_numpy(trainy.astype(int))
testx = torch.from_numpy(testx).float()
valx = torch.from_numpy(valx).float()
valy = torch.from_numpy(valy.astype(int))

# Creating NN
print('Creating NN')

hid = 200, 200
net = nn.Sequential(
    nn.Linear(40, hid[0]),
    nn.Sigmoid(),
    nn.Linear(*hid),
    nn.Sigmoid(),
    nn.Linear(hid[1], 138)
)

# Training the NN
def training_routine(net, dataset, n_iters, criterion=nn.CrossEntropyLoss(), batch_size=5000, stats_freq=10, lr=0.1):
    # organize the data
    train_data, train_labels, val_data, val_labels = dataset

    optimizer=torch.optim.SGD(net.parameters(), lr=lr)

    # GPU
    gpu = torch.cuda.is_available()
    if gpu:
        print('Using GPU')
        net = net.cuda()
    else:
        print('Not using GPU')

    print_steps = int(n_iters / stats_freq)

    # training
    for i in range(n_iters):

        train_output = []
        train_loss = []

        for j in range(0, train_data.shape[0], batch_size):
            # print progress
            if j % (batch_size * 500) == 0:
                print('\rEpoch {:4} Batch {:6} ({:.2%})'.format(i+1, j // batch_size, j / train_data.shape[0]), end='')
            
            # create batches
            b_train_data, b_train_labels = train_data[j : j + batch_size], train_labels[j : j + batch_size]

            # use gpu if possible
            if gpu:
                b_train_data, b_train_labels = b_train_data.cuda(), b_train_labels.cuda()

            # forward pass
            b_train_output = net(b_train_data)
            b_train_loss = criterion(b_train_output, b_train_labels)
            
            # backward pass and optimization
            b_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # save output
            if len(b_train_output.shape) != 0:
                x = b_train_output
                x = x.cpu().detach()
                train_output.append(x)
            if b_train_loss:
                train_loss.append(b_train_loss.cpu().detach())

        # Once every xx iterations, print statistics
        if (i+1) % print_steps == 0 or i == 0:
            # clear carriage return
            print()
            print("Computing statistics for epoch", i+1)

            # computing overall output and loss
            train_output = torch.cat(train_output)
            train_loss = torch.FloatTensor(train_loss)

            # use GPU if possible 
            if gpu:
                val_data,val_labels = val_data.cuda(),val_labels.cuda()

            # compute the accuracy of the prediction
            train_prediction = train_output.cpu().detach().argmax(dim=1)
            train_accuracy = (train_prediction.numpy()==train_labels.numpy()).mean()

            print('Prediction on training data')
            print(train_prediction)

            # Now for the validation set
            val_output = net(val_data)
            val_loss = criterion(val_output,val_labels)
            # compute the accuracy of the prediction
            val_prediction = val_output.cpu().detach().argmax(dim=1)
            val_accuracy = (val_prediction.numpy()==val_labels.cpu().detach().numpy()).mean()
            print("Training loss :",train_loss.cpu().detach().numpy())
            print("Training accuracy :",train_accuracy)
            print("Validation loss :",val_loss.cpu().detach().numpy())
            print("Validation accuracy :",val_accuracy)
            print()

            # try to save the neural network
            try:
                torch.save(net, 'neural_net')
            except:
                print('Could not save neural network.')

    net = net.cpu()

training_routine(net, (trainx, trainy, valx, valy), int(options.epochs), lr=float(options.rate), batch_size=int(options.batch_size))

try:
    torch.save(net, 'neural_net')
except:
    print('Could not save neural network.')
