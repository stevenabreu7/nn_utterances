import torch 
import numpy as np
from model import CustomNetwork

model_path = input('Enter path to model: ')

print('Loading model')
model = torch.load(model_path)

print('Loading test data')
test_data = np.load('data/test.npy', encoding='bytes')

########################################################
context = 15
padding = np.zeros((context, 40))
index_map = []
frames = []
actual_i = 0
for d in test_data:
    frames.append(padding)
    frames.append(d)
    frames.append(padding)
    # adjust index
    actual_i += context
    for _ in range(d.shape[0]):
        index_map.append(actual_i)
        actual_i += 1
    actual_i += context
test_data = np.concatenate(frames)
test_data = torch.tensor(test_data).float()
el_length = (2 * context + 1) * 40
########################################################

test_data = torch.tensor(test_data).float()

if torch.cuda.is_available():
    model = model.cuda()
    test_data = test_data.cuda()

print('Computing output')
prediction = []
for ind in index_map:
    data = test_data[ind - context : ind + context + 1].view(-1)
    output = model(test_data)
    prediction.append(output.data.max(1, keepdim = True)[1])

print('Writing results')
with open('result.txt', 'w') as f:
    f.write('id,label\n')
    for i in range(prediction.shape[0]):
        f.write('{},{}\n'.format(i, prediction[i]))
