from classes import *

# Load the data
padding = 20
pca_dim = None
trainx, trainy, valx, valy = load_training_data(padding, pca_dim)

############################################################
name = 'adam_20c_5x400_bn_64b_001'
input_n = (pca_dim or 40) * (padding * 2 + 1)
layers = [input_n, 400, 400, 400, 400, 400, 138]
batch_norm = True
epochs = 10
lrate = 1e-3
batch_size = 64
decay = False

net = generate_net(layers, batch_norm)
optimizer = torch.optim.Adam(net.parameters(), lr=lrate)
training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, optimizer=optimizer, batch_size=batch_size, decay=decay, logging=True)

############################################################
name = 'sgd_20c_5x400_bn_64b_001'
input_n = (pca_dim or 40) * (padding * 2 + 1)
layers = [input_n, 400, 400, 400, 400, 400, 138]
batch_norm = True
epochs = 10
lrate = 1e-3
batch_size = 64
decay = False

net = generate_net(layers, batch_norm)
training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, optimizer=optimizer, batch_size=batch_size, decay=decay, logging=True)
