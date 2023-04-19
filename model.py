from glob import glob
import numpy as np
import os
import random
from random import shuffle
import torch
import pandas as pd
import librosa
from torch.utils.data import Dataset
from torchvision import transforms
from hparam import hparam as hp
from librosa.core import stft, magphase
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        l1 = nn.LSTM(1024)


        def __init__(self):
            super(Model, self).__init__()
            self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
            for name, param in self.LSTM_stack.named_parameters():
              if 'bias' in name:
                 nn.init.constant_(param, 0.0)
              elif 'weight' in name:
                 nn.init.xavier_normal_(param)
            self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

        def forward(self, x):
            x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
            #only use last frame
            x = x[:,x.size(1)-1]
            x = self.projection(x.float())
            x = x / torch.norm(x, dim=1).unsqueeze(1)
            return x
