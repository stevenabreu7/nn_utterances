import numpy as np
from torch.utils.data.dataset import Dataset

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
        return X, y
    
    def __len__(self):
        return self.length
