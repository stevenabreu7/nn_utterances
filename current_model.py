from classes import *

# Load the data

trainx, trainy, valx, valy = load_training_data(20)

# network 3
name = 'adam'
layers = [410, 400, 400, 400, 400, 400, 138]
batch_norm = True
epochs = 10
lrate = 3e-1
batch_size = 64

net = generate_net(layers, batch_norm)
optimizer = torch.optim.Adam(net.parameters(), lr=lrate)
training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, optimizer=optimizer, batch_size=batch_size, decay=True, logging=True)
