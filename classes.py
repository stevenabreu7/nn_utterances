import os
import torch
import optparse
import numpy as np
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


# ### Loading data

def pad_array_temporally(X, padding):
    right_shift = lambda X, i : np.pad(X[:-i], [(i,0),(0,0)], 'constant', constant_values=0)
    left_shift = lambda X, i : np.pad(X[i:], [(0,i),(0,0)], 'constant', constant_values=0)
    before = [right_shift(X, i) for i in range(padding, 0, -1)]
    rest = [left_shift(X,i) for i in range(0, padding+1)]
    return np.concatenate(before + rest, axis=1)

def load_training_data(padding=20):
    # Getting the labels
    trainy = np.load('data/train_labels.npy', encoding='bytes')
    valy = np.load('data/dev_labels.npy', encoding='bytes')
    trainy = np.concatenate(trainy.tolist())
    valy = np.concatenate(valy.tolist())
    # PCA
    if os.path.exists('data/trainx_pca.npy'):
        trainx = np.load('data/trainx_pca.npy')
        testx = np.load('data/testx_pca.npy')
        valx = np.load('data/valx_pca.npy')
    else:
        trainx = np.load('data/train.npy', encoding='bytes')
        testx = np.load('data/test.npy', encoding='bytes')
        valx = np.load('data/dev.npy', encoding='bytes')
        trainx = np.concatenate(trainx.tolist())
        testx = np.concatenate(testx.tolist())
        valx = np.concatenate(valx.tolist())
        trainx = PCA(n_components=10).fit_transform(trainx)
        testx = PCA(n_components=10).fit_transform(testx)
        valx = PCA(n_components=10).fit_transform(valx)
        np.save('data/trainx_pca.npy', trainxs)
        np.save('data/testx_pca.npy', testxs)
        np.save('data/valx_pca.npy', valxs)
    # add context
    trainx = pad_array_temporally(trainx, padding)
    valx = pad_array_temporally(valx, padding)
    # Turn into tensors
    trainx = torch.from_numpy(trainx).float()
    trainy = torch.from_numpy(trainy.astype(int))
    testx = torch.from_numpy(testx).float()
    valx = torch.from_numpy(valx).float()
    valy = torch.from_numpy(valy.astype(int))
    # return
    return trainx, trainy, valx, valy

def training_routine(name, net, dataset, epochs, lr, optimizer=None, batch_size=5000, decay=True, logging=False):

    if logging:
        vLog = Logger('./logs/val_acc_{}'.format(name))
        tLog = Logger('./logs/train_acc_{}'.format(name))
    
    train_data, train_labels, val_data, val_labels = dataset
    
    criterion=nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    if decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    
    gpu = torch.cuda.is_available()
    print('Using GPU' if gpu else 'Not using GPU')
    
    net = net.cuda() if gpu else net
    
    for epoch in range(epochs):
        
        scheduler.step()

        train_correct = 0
        train_loss = []

        for batch_n in range(0, train_data.shape[0], batch_size):
            
            if (batch_n // batch_size) % 10 == 0:
                print('\rEpoch {:4} Batch {:6} ({:.2%})'.format(epoch + 1, batch_n // batch_size, batch_n / train_data.shape[0]), end='')
            
            a, b = batch_n, batch_n + batch_size
            batch_data, batch_labels = train_data[a:b], train_labels[a:b]

            if gpu:
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

            # forward pass
            batch_output = net(batch_data)
            batch_loss = criterion(batch_output, batch_labels)
            
            # backward pass
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            batch_prediction = batch_output.cpu().detach().argmax(dim=1)
            batch_correct = (batch_prediction.numpy() == batch_labels.cpu().detach().numpy()).sum()
            
            train_correct += batch_correct
            
            if batch_loss:
                train_loss.append(batch_loss.cpu().detach())
        
        train_accuracy = train_correct / train_data.shape[0]
        train_loss = torch.FloatTensor(train_loss)
        
        if logging:
            tLog.log_scalar('accuracy', train_accuracy, epoch + 1)
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                tLog.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                tLog.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
        
        # Once every 10 iterations, print statistics
        if True: #if epoch == 0 or epoch+1 % 10 == 0:
            
            print("\rStatistics for epoch", epoch + 1)
            
            val_loss = 0
            val_correct = 0
            count = 0
            
            for batch_n in range(0, val_data.shape[0], batch_size):
                # create batches
                a, b = batch_n, batch_n + batch_size
                batch_data = val_data[a:b]
                batch_labels = val_labels[a:b]
                
                # use GPU if possible 
                if gpu:
                    batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
                
                # Now for the validation set
                batch_output = net(batch_data)
                batch_loss = criterion(batch_output, batch_labels)
                
                # compute the accuracy of the prediction
                val_prediction = batch_output.cpu().detach().argmax(dim=1)
                val_correct += (val_prediction.numpy() == batch_labels.cpu().detach().numpy()).sum()
                
                # sum up to get mean later
                val_loss += val_loss
                count += 1
                
            # compute mean validation loss and accuracy for all batches
            val_loss = val_loss / count
            val_accuracy = val_correct / val_data.shape[0]
                
            print("Training loss :",train_loss.cpu().detach().numpy())
            print("Training accuracy :",train_accuracy)
            print("Validation loss :",val_loss)
            print("Validation accuracy :",val_accuracy)
            print()
            
            if logging:
                vLog.log_scalar('accuracy', val_accuracy, epoch + 1) 
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    tLog.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                    tLog.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            # try to save the neural network
            try:
                torch.save(net, 'neural_net_{}'.format(name))
            except:
                print('Could not save neural network.')

    net = net.cpu()

def generate_net(layers, batch_norm=True):
    model = []
    for i in range(len(layers)-1):
        model.append(nn.Linear(layers[i], layers[i+1]))
        if i < (len(layers)-2):
            if batch_norm:
                model.append(nn.BatchNorm1d(layers[i+1]))
            model.append(nn.LeakyReLU())
    net = nn.Sequential(*tuple(model))
    return net
