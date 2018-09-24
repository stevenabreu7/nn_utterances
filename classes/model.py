class Model():
    def __init__(self, name, net, epochs, criterion, optimizer, scheduler, logging):
        self.name = name 
        self.net = net 
        self.epochs = epochs 
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.logging = logging