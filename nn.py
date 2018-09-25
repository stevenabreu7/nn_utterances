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
NUM_PHONEMES = 138

def load_data(context):
    train_x = np.load('data/train.npy', encoding='bytes')
    train_y = np.load('data/train_labels.npy', encoding='bytes')
    train = PhonemeDataset(train_x, train_y, context)
    dev_x = np.load('data/dev.npy', encoding='bytes')
    dev_y = np.load('data/dev_labels.npy', encoding='bytes')
    dev = PhonemeDataset(dev_x, dev_y, context)
    return train, dev


class PhonemeDataset(Dataset):
    def __init__(self, utterances, utterance_labels, context):
        self.context = context
        padding = np.array([[0] * FRAME_LEN for _ in range(context)])

        if self.context:
            frames = []
            current_frame = 0
            self._idx_map = []
            for utterance in utterances:
                frames.append(padding)
                frames.append(utterance)
                frames.append(padding)

                current_frame += self.context
                for _ in range(utterance.shape[0]):
                    self._idx_map.append(current_frame)
                    current_frame += 1
                current_frame += self.context
            self.frames = np.concatenate(frames, axis=0)
        else:
            self.frames = np.concatenate(utterances)
            self._idx_map = list(range(self.frames.shape[0]))

        self.frames = torch.tensor(self.frames).float()
        self.labels = torch.tensor(np.concatenate(utterance_labels))
        self.instance_size = (2 * self.context + 1) * FRAME_LEN

    def __len__(self):
        # changed
        return self.labels.shape[0]

    def __getitem__(self, idx):
        idx_left = self._idx_map[idx] - self.context
        idx_right = self._idx_map[idx] + self.context + 1
        
        instance = self.frames[idx_left:idx_right].view(-1)

        label = self.labels[idx]

        return (instance, label)

class PhonemeModel(nn.Module):
    
    def __init__(self, context):
        super(PhonemeModel, self).__init__()
        self.input_size = (2 * context + 1) * FRAME_LEN
        self.output_size = NUM_PHONEMES
        
        self.layer1 = nn.Linear(self.input_size, 1024)
        self.layer1bn = nn.modules.BatchNorm1d(1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer2bn = nn.modules.BatchNorm1d(1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer3bn = nn.modules.BatchNorm1d(1024)
        self.layer4 = nn.Linear(1024, 512)
        self.layer4bn = nn.modules.BatchNorm1d(512)
        self.layer5 = nn.Linear(512, 256)
        self.layer5bn = nn.modules.BatchNorm1d(256)
        self.layer6 = nn.Linear(256, 256)
        self.layer6bn = nn.modules.BatchNorm1d(256)
        self.layer7 = nn.Linear(256, self.output_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1bn(x)
        x = F.leaky_relu(x)
        x = self.layer2(x)
        x = self.layer2bn(x)
        x = F.leaky_relu(x)
        x = self.layer3(x)
        x = self.layer3bn(x)
        x = F.leaky_relu(x)
        x = self.layer4(x)
        x = self.layer4bn(x)
        x = F.leaky_relu(x)
        x = self.layer5(x)
        x = self.layer5bn(x)
        x = F.leaky_relu(x)
        x = self.layer6(x)
        x = self.layer6bn(x)
        x = F.leaky_relu(x)
        x = self.layer7(x)
        return x

def inference(model, loader, gpu):
    num_predicted = 0
    correct = 0

    for data, label in loader:
        X = Variable(data.view(-1, loader.dataset.instance_size))
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

                X = Variable(data.view(-1, train_loader.dataset.instance_size))
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
                    print(correct / num_predicted)

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
instance_size = train.instance_size

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
