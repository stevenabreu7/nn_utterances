import torch
from classes.model import Model
from classes.trainer import Trainer

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