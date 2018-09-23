from classes import *

# Load the data

trainx, trainy, valx, valy = load_training_data(20)

# # network 1
# name = 'id1'
# layers = [410, 400, 400, 400, 138]
# batch_norm = True
# epochs = 40
# lrate = 3e-1
# batch_size = 5000
# decay = True

# net = generate_net(layers, batch_norm)
# training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, batch_size=batch_size, decay=True, logging=True)

# # network 2
# name = 'id2'
# layers = [410, 400, 400, 400, 400, 138]
# batch_norm = True
# epochs = 40
# lrate = 3e-1
# batch_size = 5000
# decay = True

# net = generate_net(layers, batch_norm)
# training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, batch_size=batch_size, decay=True, logging=True)

# network 3
name = 'id3'
layers = [410, 400, 400, 400, 400, 400, 138]
batch_norm = True
epochs = 40
lrate = 3e-1
batch_size = 5000
decay = True

net = generate_net(layers, batch_norm)
training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, batch_size=batch_size, decay=True, logging=True)

# network 4
name = 'id4'
layers = [410, 2000, 1000, 1000, 500, 500, 250, 138]
batch_norm = True
epochs = 40
lrate = 3e-1
batch_size = 5000
decay = True

net = generate_net(layers, batch_norm)
training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, batch_size=batch_size, decay=True, logging=True)

# network 5
name = 'id5'
layers = [410, 400, 400, 400, 138]
batch_norm = True
epochs = 40
lrate = 3e-1
batch_size = 500
decay = True

net = generate_net(layers, batch_norm)
training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, batch_size=batch_size, decay=True, logging=True)

# network 6
name = 'id6'
layers = [410, 800, 800, 400, 400, 400, 200, 200, 138]
batch_norm = True
epochs = 40
lrate = 3e-1
batch_size = 5000
decay = True

net = generate_net(layers, batch_norm)
training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, batch_size=batch_size, decay=True, logging=True)

# network 7
name = 'id7'
layers = [410, 800, 800, 400, 400, 400, 200, 200, 138]
batch_norm = True
epochs = 40
lrate = 3e-1
batch_size = 64
decay = True

net = generate_net(layers, batch_norm)
training_routine(name, net, (trainx, trainy, valx, valy), epochs, lrate, batch_size=batch_size, decay=True, logging=True)