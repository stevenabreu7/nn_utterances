# general imports
import torch
# library imports
from torch.utils.data import DataLoader
# first party imports
from logger import Logger
from customDataset import CustomDataset

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
