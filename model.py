import torch 
import numpy as np 
import torch.nn as nn
from torch.functional import F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.autograd.variable import Variable

LEN_FRAME = 40
LEN_PHONEME = 138

class CustomDataset(Dataset):
    def __init__(self, data, labels, context):
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

        # index mapping for retrieving items
        self.index_map = []
        # list of all frames, will be concatenated
        frames = []
        # keep track of actual index
        actual_i = 0
        for d in data:
            # add padding before and after each utterance
            frames.append(padding)
            frames.append(d)
            frames.append(padding)
            # adjust index
            actual_i += self.context
            for _ in range(d.shape[0]):
                self.index_map.append(actual_i)
                actual_i += 1
            actual_i += self.context

        # save data as array with proper dimensions
        self.data = np.concatenate(frames)
        self.data = torch.tensor(self.data).float()
        # save labels as 1d array
        self.labels = np.concatenate(labels)
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

def load_data(context):
    """ Return the train and val dataset, with a certain context.
    """
    train_data = np.load('data/train.npy', encoding='bytes')
    train_labels = np.load('data/train_labels.npy', encoding='bytes')
    val_data = np.load('data/dev.npy', encoding='bytes')
    val_labels = np.load('data/dev_labels.npy', encoding='bytes')
    train_dataset = CustomDataset(train_data, train_labels, context)
    val_dataset = CustomDataset(val_data, val_labels, context)
    return train_dataset, val_dataset

class CustomNetwork(nn.Module):
    def __init__(self, context):
        """ Custom network, takes context as argument.
        Architecture: 
            1000 x 1000 x 1000 x 500 x 250 x 250
        """
        super(CustomNetwork, self).__init__()
        # input and output size
        input_n = (2 * context + 1) * LEN_FRAME
        output_n = LEN_PHONEME
        # layers
        self.layer1 = nn.Linear(input_n, 1000)
        self.layer1b = nn.modules.BatchNorm1d(1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer2b = nn.modules.BatchNorm1d(1000)
        self.layer3 = nn.Linear(1000, 1000)
        self.layer3b = nn.modules.BatchNorm1d(1000)
        self.layer4 = nn.Linear(1000, 500)
        self.layer4b = nn.modules.BatchNorm1d(500)
        self.layer5 = nn.Linear(500, 250)
        self.layer5b = nn.modules.BatchNorm1d(250)
        self.layer6 = nn.Linear(250, 250)
        self.layer6b = nn.modules.BatchNorm1d(250)
        self.layer7 = nn.Linear(250, output_n)
    
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

class Trainer:
    def __init__(self, train_loader, val_loader, name, net, optimizer, criterion, scheduler):
        print('Loading Trainer class for {}. '.format(name))
        # save the loaders
        self.update_data(train_loader, val_loader)
        # update training model
        self.update_model(name, net, optimizer, criterion, scheduler)
        # check GPU availability
        self.gpu = torch.cuda.is_available()
        print('Using GPU' if self.gpu else 'Not using GPU')
    
    def save_model(self):
        torch.save(self.net.state_dict(), 'models/{}'.format(self.name))
    
    def update_data(self, train_loader, val_loader):
        self.val_loader = val_loader
        self.train_loader = train_loader
    
    def update_model(self, name, net, optimizer, criterion, scheduler):
        self.net = net
        self.name = name 
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
    
    def train(self, epochs):
        print('Start training {}.'.format(self.name))

        # move network to GPU if possible
        self.net = self.net.cuda() if self.gpu else self.net 

        for epoch in range(epochs):

            if scheduler:
                scheduler.step()

            ##############################
            # TRAINING DATA
            ##############################
            
            train_num = 0
            train_loss = 0
            train_correct = 0

            for batch_i, (batch_data, batch_labels) in enumerate(self.train_loader):
                
                # reset optimizer gradients to zero
                self.optimizer.zero_grad()

                # initialize the data as variables
                batch_data = Variable(batch_data.view(-1, self.train_loader.dataset.el_length))
                batch_labels = Variable(batch_labels)

                # move data to GPU if possible
                batch_data = batch_data.cuda() if self.gpu else batch_data
                batch_labels = batch_labels.cuda() if self.gpu else batch_labels

                # forward pass of the data
                batch_output = self.net(batch_data)

                # evaluate the prediction and correctness
                batch_prediction = batch_output.data.max(1, keepdim = True)[1]
                batch_prediction = batch_prediction.eq(batch_labels.data.view_as(batch_prediction))
                train_correct += batch_prediction.sum()
                train_num += batch_data.data.shape[0]

                # compute the losss
                batch_loss = self.criterion(batch_output, batch_labels)
                
                # backward pass and optimizer step
                batch_loss.backward()
                self.optimizer.step()

                # sum up this batch's loss
                train_loss += batch_loss.data.item()

                # print training progress
                if batch_i % 10 == 0:
                    print('\rEpoch {:3} Progress {:5.2%} Accuracy {:5.2%}'.format(
                        epoch + 1, 
                        batch_i * self.train_loader.batch_size / len(self.train_loader.dataset),
                        train_correct.cpu().item() / ((batch_i + 1) * self.train_loader.batch_size)
                    ), end='')

            # compute epoch loss and accuracy
            train_loss = train_loss / train_num
            train_accuracy = train_correct.cpu().item() / train_num

            # print summary for this epoch
            print('\rEpoch {:3} finished.\t\t\t\nTraining Accuracy: {:5.2%}\nTraining Loss: {:10.7f}'.format(
                epoch + 1, 
                train_accuracy, 
                train_loss
            ))

            ##############################
            # VALIDATION DATA
            ##############################

            val_num = 0
            val_loss = 0
            val_correct = 0

            for batch_i, (batch_data, batch_labels) in enumerate(self.val_loader):

                # initialize the data as variables
                batch_data = Variable(batch_data.view(-1, self.train_loader.dataset.el_length))
                batch_labels = Variable(batch_labels)

                # move data to GPU if possible
                batch_data = batch_data.cuda() if self.gpu else batch_data
                batch_labels = batch_labels.cuda() if self.gpu else batch_labels

                # forward pass of the data
                batch_output = self.net(batch_data)

                # evaluate the prediction and correctness
                batch_prediction = batch_output.data.max(1, keepdim = True)[1]
                batch_prediction = batch_prediction.eq(batch_labels.data.view_as(batch_prediction))
                val_correct += batch_prediction.sum()
                val_num += batch_data.data.shape[0]

                # compute the losss
                batch_loss = self.criterion(batch_output, batch_labels)

                # sum up this batch's loss
                val_loss += batch_loss.data.item()

            # compute validation loss and accuracy
            val_loss = val_loss / val_num
            val_accuracy = val_correct.cpu().item() / val_num

            # print validation stats
            print('Validation Accuracy: {:5.2%}\nValidation Loss: {:10.7f}'.format(
                val_accuracy, 
                val_loss
            ))

            torch.save(self.net, 'models/{}_{}'.format(self.name, epoch))

        # move network back to CPU if needed
        self.net = self.net.cpu() if self.gpu else self.net 

def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0,1)

def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)

if __name__ == '__main__':
    # data parameters
    context = 15
    batch_size = 10000

    # datasets and loaders
    print('Loading datasets')
    train_dataset, val_dataset = load_data(context)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=RandomSampler(val_dataset))

    # model
    net = CustomNetwork(context)
    net.apply(init_xavier)

    # training parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
    criterion = nn.modules.loss.CrossEntropyLoss()

    # initialize the trainer
    trainer = Trainer(train_loader, val_loader, 'lr_decay', net, optimizer, criterion, scheduler)

    # run the training
    epochs = 10
    trainer.train(epochs)
    trainer.save_model()