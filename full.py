import torch
import numpy as np
import tensorflow as tf
from torch.functional import F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class Trainer():
    def __init__(self, trainx, trainy, valx, valy, padding, context, batch_size):
        self.input_dim = 40 * (2 * padding + 1)
        self.output_dim = 138
        print('Loading datasets', end='')
        train_dataset = CustomDataset('data/train.npy', 'data/train_labels.npy', padding, context)
        val_dataset = CustomDataset('data/dev.npy', 'data/dev_labels.npy', padding, context)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        print('\rLoaded datasets.   ')
        
    def training_routine(self, model):
        """ Train a neural network.
        Model parameters:
            name            Name of the model that is to be trained
            net             Pytorch network for this model
            train_loader    DataLoader class instance with the training data
            val_loader      DataLoader class instance with the validation data
            epochs          Number of epochs to run
            criterion       Criterion to use (recommended: torch.nn.CrossEntropyLoss() )
            optimizer       Optimizer to use (recommended: SGD or ADAM)
            scheduler       Scheduler to use (optional, no scheduler will be used if None)
            logging         True/False, whether or not to log to Tensorboard
        """
        name = model.name
        net = model.net 
        epochs = model.epochs 
        criterion = model.criterion
        optimizer = model.optimizer
        scheduler = model.scheduler
        logging = model.logging

        # initialize tensorboard logging
        if logging:
            vLog = Logger('./logs/val_acc_{}'.format(name))
            tLog = Logger('./logs/train_acc_{}'.format(name))
        
        # check GPU availability
        gpu = torch.cuda.is_available()
        print('Using GPU' if gpu else 'Not using GPU')
        
        # move network to GPU if possible
        net = net.cuda() if gpu else net
        
        for epoch in range(epochs):

            # learning rate decay (optional)
            if scheduler:
                scheduler.step()

            # performance metrics for training set
            train_correct = 0
            train_count = 0
            train_loss = []

            # train the network
            for batch_n, (batch_data, batch_labels) in enumerate(self.train_loader):
                
                print('\rTraining epoch {:4} Batch {:6} ({:.2%})'.format(epoch + 1, batch_n, batch_n / len(self.train_loader)), end=' ' * 10)

                if gpu:
                    batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

                # forward pass
                batch_output = net(batch_data)
                batch_loss = criterion(batch_output, batch_labels)
                
                # backward pass
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # performance for this batch
                batch_prediction = batch_output.cpu().detach().argmax(dim=1)
                batch_correct = (batch_prediction.numpy() == batch_labels.cpu().detach().numpy()).sum()
                
                # track overall epoch performance
                train_correct += batch_correct
                train_count += batch_data.shape[0]
                train_loss.append(batch_loss.cpu().detach())
            
            train_accuracy = train_correct / train_count

            # print statistics
            print('\nValidation epoch {:4}'.format(epoch + 1))
            
            # performance metrics for validation set
            val_loss = []
            val_correct = 0
            val_count = 0
            
            for batch_n, (batch_data, batch_labels) in enumerate(self.val_loader):
                
                if gpu:
                    batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
                
                # forward pass
                batch_output = net(batch_data)
                batch_loss = criterion(batch_output, batch_labels)
                
                # performance for this batch
                batch_prediction = batch_output.cpu().detach().argmax(dim=1)
                batch_correct = (batch_prediction.numpy() == batch_labels.cpu().detach().numpy()).sum()

                # track overall epoch performance
                val_correct += batch_correct
                val_count += batch_data.shape[0]
                val_loss.append(batch_loss.cpu().detach())

            val_accuracy = val_correct / val_count
                
            # print the statistics
            print("Training accuracy: {:.4f}".format(train_accuracy))
            print("Validation accuracy:{:.4f}".format(val_accuracy))
            print()
            
            # log data to tensorboard
            if logging:
                vLog.log_scalar('accuracy', val_accuracy, epoch + 1) 
                tLog.log_scalar('accuracy', train_accuracy, epoch + 1)
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    tLog.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                    tLog.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            # save the neural network
            try:
                torch.save(net, 'models/neural_net_{}'.format(name))
            except:
                print('Could not save neural network.')
        
        # move the network back to CPU
        net = net.cpu()

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : Name of the scalar
        value : value itself
        step :  training iteration
        """
        # Notice we're using the Summary "class" instead of the "tf.summary" public API.
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

class CustomDataset(Dataset):
    def __init__(self, data_file, label_file, padding, context):
        # save the context and padding
        self.padding = padding
        self.context = context
        # load data
        self.X = np.load(data_file, encoding='bytes')
        self.y = np.load(label_file, encoding='bytes')
        # padding for each utterance
        pad = np.zeros((self.padding, self.X[0].shape[1]))
        # map for actual indices
        self.index_map = []
        # keep track of actual index
        actual_i = 0
        for i in range(self.X.shape[0]):
            # add padding to utterance
            self.X[i] = np.concatenate([pad, self.X[i], pad])
            # map the indices correctly
            actual_i += self.context
            for j in range(self.X[i].shape[0]):
                self.index_map.append(actual_i)
                actual_i += 1
            actual_i += self.context
        # flatten array to list of frames
        self.X = np.concatenate(self.X.tolist())
        self.X = torch.tensor(self.X).float()
        self.y = np.concatenate(self.y.tolist())
        # save length
        self.length = self.y.shape[0]
    
    def __getitem__(self, index):
        # getting y
        y = self.y[index]
        # determining index for X
        a = self.index_map[index] - self.context
        b = self.index_map[index] + self.context + 1
        # getting X
        X = self.X[a:b].view(-1)
        # return it
        return X, y
    
    def __len__(self):
        return self.length

class Model():
    def __init__(self, name, net, epochs, criterion, optimizer, scheduler, logging):
        self.name = name 
        self.net = net 
        self.epochs = epochs 
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.logging = logging

def init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)

# general initialization
padding, context = 15, 15
batch_size = 10000
trainer = Trainer('data/train.npy', 'data/train_labels.npy', 'data/dev.npy', 'data/dev_labels.npy', padding, context, batch_size)

# model
class CustomNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 1000)
        self.layer1b = torch.nn.modules.BatchNorm1d(1000)
        self.layer2 = torch.nn.Linear(1000, 1000)
        self.layer2b = torch.nn.modules.BatchNorm1d(1000)
        self.layer3 = torch.nn.Linear(1000, 1000)
        self.layer3b = torch.nn.modules.BatchNorm1d(1000)
        self.layer4 = torch.nn.Linear(1000, 500)
        self.layer4b = torch.nn.modules.BatchNorm1d(500)
        self.layer5 = torch.nn.Linear(500, 250)
        self.layer5b = torch.nn.modules.BatchNorm1d(250)
        self.layer6 = torch.nn.Linear(250, 250)
        self.layer6b = torch.nn.modules.BatchNorm1d(250)
        self.layer7 = torch.nn.Linear(250, output_size)
        
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

# model - 1000x1000x1000x500x250x250
name = 'test_model'
net = CustomNetwork(trainer.input_dim, trainer.output_dim)
epochs = 10
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
scheduler = None
logging = True
# save it into a model
model = Model(name, net, epochs, criterion, optimizer, scheduler, logging)
# train the model
trainer.training_routine(model)