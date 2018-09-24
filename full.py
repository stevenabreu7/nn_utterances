import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class Trainer():
    def __init__(self, trainx, trainy, valx, valy, padding, context, batch_size):
        self.input_dim = 40 * (2 * padding + 1)
        self.output_dim = 138
        train_dataset = CustomDataset('data/train.npy', 'data/train_labels.npy', padding, context)
        val_dataset = CustomDataset('data/dev.npy', 'data/dev_labels.npy', padding, context)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
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
            train_accuracy = 0
            train_loss = []

            # train the network
            for batch_n, (batch_data, batch_labels) in enumerate(self.train_loader):
                
                if batch_n % 100 == 0:
                    print('\rTraining epoch {:4} Batch {:6} ({:.2%})'.format(epoch + 1, batch_n, batch_n / len(self.train_loader)), end=' ' * 10)

                if gpu:
                    batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

                # forward pass
                print(batch_data.shape)
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
                train_accuracy += batch_correct / len(self.train_loader.dataset)
                train_loss.append(batch_loss.cpu().detach())
            
            # print statistics
            print('\rValidation epoch {:4}'.format(epoch + 1), end=' ' * 30)
            
            # performance metrics for validation set
            val_loss = []
            val_accuracy = 0
            
            for batch_n, (batch_data, batch_labels) in enumerate(self.val_loader):
                
                if gpu:
                    batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
                
                # forward pass
                batch_output = net(batch_data)
                batch_loss = criterion(batch_output, batch_labels)
                
                # performance for this batch
                batch_prediction = batch_output.cpu().detach().argmax(dim=1)
                batch_correct += (batch_prediction.numpy() == batch_labels.cpu().detach().numpy()).sum()

                # track overall epoch performance
                val_accuracy += batch_correct / len(self.val_loader.dataset)
                val_loss.append(batch_loss.cpu().detach())
                
            # print the statistics
            print("Training loss :",train_loss)
            print("Training accuracy :",train_accuracy)
            print("Validation loss :",val_loss)
            print("Validation accuracy :",val_accuracy)
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
        # get length
        self.length = sum([e.shape[0] for e in self.X])
        # helper list for determining index
        self.lol = np.array([e.shape[0] for e in self.X]).cumsum()
        # pad data with zeros between utterances
        for i in range(self.X.shape[0]):
            pad = np.zeros((self.padding, self.X[i].shape[1]))
            self.X[i] = np.concatenate([pad, self.X[i], pad])
        # flatten array to list of frames, separated by zeros
        self.X = np.concatenate(self.X.tolist())
        self.y = np.concatenate(self.y.tolist())
    
    def __getitem__(self, index):
        # getting y
        y = self.y[index]
        # determining index for X
        index += self.padding
        i = 0
        while i < len(self.lol) and index >= self.lol[i]:
            index += 2 * self.padding
            i += 1
        # getting X
        X = self.X[index - self.context : index + self.context + 1]
        X = np.concatenate(X.tolist())
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y.astype(int))
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

# general initialization
padding, context = 10, 10
batch_size = 5000
trainer = Trainer('data/train.npy', 'data/train_labels.npy', 'data/dev.npy', 'data/dev_labels.npy', padding, context, batch_size)

# model - 200x200x200
name = 'test_model'
net = torch.nn.Sequential(
    torch.nn.Linear(trainer.input_dim, 200),
    torch.nn.BatchNorm1d(200),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.BatchNorm1d(200),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.BatchNorm1d(200),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(200, trainer.output_dim)
)
epochs = 10
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
scheduler = None
logging = True
# save it into a model
model = Model(name, net, epochs, criterion, optimizer, scheduler, logging)
# train the model
trainer.training_routine(model)