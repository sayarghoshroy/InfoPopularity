import csv
import torch
import torch.utils.data as data
from torch.autograd import Variable
from .Vocab import Vocab
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, examples):
        super(Dataset,self).__init__()
        self.examples = examples 
        self.training = False
    def train(self):
        self.training = True
        return self
    def test(self):
        self.training = False
        return self
    def shuffle(self,words):
        np.random.shuffle(words)
        return ' '.join(words)
    def dropout(self,words,p=0.3):
        l = len(words)
        drop_index = np.random.choice(l,int(l*p))
        keep_words = [words[i] for i in range(l) if i not in drop_index]
        return ' '.join(keep_words)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex
    def __len__(self):
        return len(self.examples)
