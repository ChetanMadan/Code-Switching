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

class CodeSwitchDataset(Dataset):

    def __init__(self, lang, mode = "train", shuffle=True):
        self.mode = mode
        # data path
        self.lang = lang
        if self.lang == "Gujarati":
            self.max_len = 0
        elif self.lang == "Telugu":
            self.max_len = 529862
        elif self.max_len == 'Tamil':
            self.max_len = 0
        else:
            raise Exception("Check Language")
        if self.mode == "train":
            self.path = 'Data/PartA_{}/Train/'.format(self.lang)
        elif self.mode == "test":
            self.path = self.path = 'Data/PartA_{}/Dev/'.format(self.lang)
        else:
            raise Exception("Incorrect mode")
        self.file_list = os.listdir(os.path.join(self.path, 'Audio'))
        self.shuffle=shuffle
        self.csv_file = pd.read_csv(self.path + 'Transcription_LT_Sequence.tsv', header=None, sep='\t')

    def __len__(self):
        return len(self.csv_file)

    def pad(self, wav, max_len):
        while len(wav) < max_len:
            diff = max_len - len(wav)
            ext = wav[:diff]
            wav = np.append(wav, wav[:diff])
        return wav

    def preprocess(self, wav, sr):
        out = stft(wav, win_length=int(sr*0.02), hop_length=int(sr*0.01))
        out = magphase(out)[0]
        out = [np.log(1 + x) for x in out]
        return np.array(out)

    def __getitem__(self, idx):
        file_name = self.csv_file[0][idx]
        trans = self.csv_file[1][idx]
        wav, sr = librosa.load(glob(self.path + 'Audio/*'+ str(file_name) + '.wav')[0])
        wav = self.pad(wav, self.max_len)
        out = self.preprocess(wav, sr)
        out = np.transpose(out, axes=(1, 0))

        if len(set(trans)) > 2:
            label = 1
        elif len(set(trans)) == 1 or len(set(trans)) == 2:
            label = 0
        else:
            raise Exception("Check transcript")
        if self.mode =="train":
            return out, label
        elif self.mode == "test":
            return out
        else:
            raise Exception("Incorrect Mode")

