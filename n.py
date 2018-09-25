import torch
import torch.nn as nn
import numpy as np
import os.path
import sys
import time
import hashlib
from collections import namedtuple
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.functional import F

FRAME_LEN = 40
LEN_FRAME = 40
NUM_PHONEMES = 138
LEN_PHONEME = 138

class CustomDataset(Dataset):
    def __init__(self, utterances_path, labels_path, context, pca=None):
        """ Class that contains the dataset for this task.

        The input data is a list of k utterances, each containing n_k 
        frames, with each frame having 40 dimensions.
        The input labels is a list of k utterances, each containing n_k 
        labels, with integers between 0 and 137.
        
        The resulting data is a flat list of all frames, separated by 
        x zeros, with x being the context.
        """
        self.context = context 

        # padding for each utterance
        padding = np.zeros((self.context, LEN_FRAME))

        # load data from files
        self.data = np.load(utterances_path, encoding='bytes')
        self.labels = np.load(labels_path, encoding='bytes')

        # index mapping for retrieving items
        self.index_map = []
        actual_i = 0

        for i in range(self.data.shape[0]):
            # pad each utterance with zeros before
            self.data[i] = np.concatenate([padding, self.data[i]])
            # adjust index
            actual_i += self.context
            for _ in range(self.data[i].shape[0]):
                self.index_map.append(actual_i)
                actual_i += 1
        # pad after the last instance as well
        self.data[i] = np.concatenate([self.data[i], padding])
        
        # save data as array with proper dimensions
        self.data = np.concatenate(self.data)
        self.data = torch.tensor(self.data).float()
        # save labels as 1d array
        self.labels = np.concatenate(self.labels)
        self.labels = torch.tensor(self.labels)
        
        # save the length of each data point
        self.el_length = (2 * self.context + 1) * LEN_FRAME
    
    def __len__(self):
        """ Get the number of instances in the dataset.
        (i.e. the number of labels)
        """
        return self.labels.shape[0]

    def __getitem__(self, i):
        """ Return the i-th frame and label of the dataset.
        We have to account for the padding.
        """
        a = self.index_map[i] - self.context
        b = self.index_map[i] + self.context
        X = self.data[a : b+1].view(-1)
        y = self.labels[i]
        return X, y

def load_data(context, pca=None):
    """ Return the train and val dataset, with a certain context.
    """
    train_dataset = CustomDataset('data/train.npy', 'data/train_labels.npy', context)
    val_dataset = CustomDataset('data/dev.npy', 'data/dev_labels.npy', context)
    return train_dataset, val_dataset

class PhonemeModel(nn.Module):
    def __init__(self, context):
        """ Custom network, takes context as argument.
        Architecture: 
            1000 x 1000 x 1000 x 500 x 250 x 250
        """
        super(PhonemeModel, self).__init__()
        # input and output size
        input_n = (2 * context + 1) * LEN_FRAME
        output_n = LEN_PHONEME
        # layers
        self.layer1 = nn.Linear(input_n, 1024)
        self.layer1b = nn.modules.BatchNorm1d(1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer2b = nn.modules.BatchNorm1d(1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer3b = nn.modules.BatchNorm1d(1024)
        self.layer4 = nn.Linear(1024, 512)
        self.layer4b = nn.modules.BatchNorm1d(512)
        self.layer5 = nn.Linear(512, 256)
        self.layer5b = nn.modules.BatchNorm1d(256)
        self.layer6 = nn.Linear(256, 256)
        self.layer6b = nn.modules.BatchNorm1d(256)
        self.layer7 = nn.Linear(256, output_n)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1b(x)
        x = F.leaky_relu(x)
        x = self.layer2(x)
        x = self.layer2b(x)
        x = F.leaky_relu(x)
        x = self.layer3(x)
        x = self.layer3b(x)
        x = F.leaky_relu(x)
        x = self.layer4(x)
        x = self.layer4b(x)
        x = F.leaky_relu(x)
        x = self.layer5(x)
        x = self.layer5b(x)
        x = F.leaky_relu(x)
        x = self.layer6(x)
        x = self.layer6b(x)
        x = F.leaky_relu(x)
        x = self.layer7(x)
        return x 

def inference(model, loader, gpu):
    num_predicted = 0
    correct = 0

    for data, label in loader:
        X = Variable(data.view(-1, loader.dataset.el_length))
        Y = Variable(label)
        if gpu:
            X = X.cuda()
            Y = Y.cuda()
        out = model(X)

        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
        num_predicted += X.data.shape[0]

    return correct.item() / num_predicted


class Trainer:
    def __init__(self, model, optimizer, criterion, gpu, load_path=None):
        self.model = model
        self.gpu = gpu
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer
        self.criterion = criterion

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, epochs, train_loader, dev_loader):
        self.metrics = []
        
        if self.gpu:
            self.model = self.model.cuda()

        for epoch in range(epochs):

            self.model.train()
            epoch_loss = 0
            correct = 0
            num_predicted = 0

            for batch_idx, (data, label) in enumerate(train_loader):

                self.optimizer.zero_grad()

                X = Variable(data.view(-1, train_loader.dataset.el_length))
                Y = Variable(label)
                if self.gpu:
                    X = X.cuda()
                    Y = Y.cuda()

                out = self.model(X)

                pred = out.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                
                correct += predicted.sum()
                num_predicted += X.data.shape[0]

                loss = self.criterion(out, Y)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

                if batch_idx % 10 == 0 and batch_idx > 0:
                    print('progress: {:7.3f}% correct: {:7d} accuracy: {:2.3f}%'.format(
                        batch_idx * train_loader.batch_size * 100 / len(train_loader.dataset),
                        correct.cpu().item(),
                        correct.cpu().item() * 100 / (batch_idx * train_loader.batch_size)
                    ))

            self.model.eval()
            total_loss = epoch_loss / num_predicted
            train_arccuracy = correct.item() / num_predicted
            val_accuracy = inference(self.model, dev_loader, self.gpu)

            print('')
            print("epoch: {}, loss: {:.8f}, train_acc: {:8.3f}%, val_acc: {:8.3f}%,".format(
                epoch + 1, total_loss, train_arccuracy * 100, val_accuracy * 100))

def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0,1)
        
def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)

context_frames = 14
train, dev = load_data(context=context_frames)

batch_size = 10000
train_loader = DataLoader(train, batch_size=batch_size, sampler=RandomSampler(train))
dev_loader = DataLoader(dev, batch_size=batch_size, sampler=RandomSampler(dev))

model = PhonemeModel(context_frames)
model.apply(init_xavier)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
criterion = nn.modules.loss.CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion, torch.cuda.is_available())

trainer.run(5, train_loader, dev_loader)
trainer.save_model('models/bla')
