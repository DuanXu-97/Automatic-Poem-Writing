import numpy as np
import torch
from torch.utils import data


class PoemDataset(data.Dataset):

    def __init__(self, data_path, config):

        datas = np.load(data_path)
        data = datas['data']
        self.data = torch.from_numpy(data)
        self.ix2word = datas['ix2word'].item()
        self.word2ix = datas['word2ix'].item()
        self.config = config

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = np.load("data/tang.npz")
    data = dataset['data']
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()
    print(data[0])



