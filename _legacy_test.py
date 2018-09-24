import torch 
import optparse
import numpy as np
from sklearn.decomposition import PCA
from classes import pad_array_temporally

# Parse options
parser = optparse.OptionParser()

parser.add_option('-n', '--net', action="store", dest="net", help="path to the model")

options, args = parser.parse_args()

# loading data
print('loading data')
testx = np.load('data/test.npy', encoding='bytes')
testx = np.concatenate(testx.tolist())
testx = PCA(n_components=10).fit_transform(testx)
padding = 20
testx = pad_array_temporally(testx, padding)
testx = torch.from_numpy(testx).float()

# loading network
print('loading network')
net = torch.load(options.net, map_location='cpu')

# calculating
print('calculating result')
output = net(testx)
pred = output.cpu().detach().argmax(dim=1)

# saving it
with open('result.txt', 'w') as f:
    f.write('id,label\n')
    for i in range(len(pred)):
        f.write('{},{}\n'.format(i, pred[i]))
