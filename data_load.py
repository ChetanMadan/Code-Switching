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
class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        s 0
        e 1
        t 2
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

class CodeSwitchDataset(Dataset):
    def __init__(self, lang, mode = "train", shuffle=True):
        self.mode = mode
        # data path
        self.lang = lang
        if self.lang == "Gujarati":
            self.max_len = 0
        elif self.lang == "Telugu":
            self.max_len = 529862
        elif self.lang == 'Tamil':
            self.max_len = 0
        else:
            raise Exception("Check Language")
        if self.mode == "train":
            self.path = 'Data/PartA_{}/Train/'.format(self.lang)
        elif self.mode == "test":
            self.path = self.path = 'Data/PartB_{}/Dev/'.format(self.lang)
        else:
            raise Exception("Incorrect mode")
        self.file_list = os.listdir(os.path.join(self.path, 'Audio'))
        self.shuffle=shuffle
        self.csv_file = pd.read_csv(self.path + 'Transcription_LT_Sequence.tsv', header=None, sep='\t')
        self.input_length = []
        self.label_length = []

    def __len__(self):
        return len(self.csv_file)

    def pad(self, wav, trans, max_len):
        while len(wav) < max_len:
            diff = max_len - len(wav)
            ext = wav[:diff]
            wav = np.append(wav, wav[:diff])
            ratio = int(len(trans)*diff/len(wav))
            trans +=trans[:ratio]
        return wav, trans

    def preprocess(self, wav, sr, trans):

        out = stft(wav, win_length=int(sr*0.02), hop_length=int(sr*0.01))
        text_transform = TextTransform()
        trans = torch.Tensor(text_transform.text_to_int(trans.lower()))

        out = magphase(out)[0]
        out = [np.log(1 + x) for x in out]
        return np.array(out), trans

    def __getitem__(self, idx):
        file_name = self.csv_file[0][idx]
        trans = self.csv_file[1][idx]
        wav, sr = librosa.load(glob(self.path + 'Audio/*'+ str(file_name) + '.wav')[0])
        #wav, trans  = self.pad(wav, trans, self.max_len)

        #out, trans = self.preprocess(wav, sr, trans)


        if len(set(trans)) > 2:
            label = 1
        elif len(set(trans)) == 1 or len(set(trans)) == 2:
            label = 0
        else:
            raise Exception("Check transcript")
        if self.mode =="train":
            return wav, sr, trans, self.lang
        elif self.mode == "test":
            return wav
        else:
            raise Exception("Incorrect Mode")
