import librosa
import torch
import torchaudio
import torch.optim as optim

import numpy as np
from difflib import SequenceMatcher
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import torch
import torch.nn as nn
from librosa.core import stft, magphase
from torch.autograd import Variable
#from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        s 1
        e 2
        t 3
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string)


train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(n_mels=128, sample_rate = 22050, n_fft = 512, win_length=int(22050*0.02), hop_length=int(22050*0.01)),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )






valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _) in data:
        waveform=torch.Tensor(waveform)
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, padding_value = 1, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths



BATCH_SIZE = 4

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation

    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)]

def GreedyDecoder(output, labels, label_lengths, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 64, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[

            ResidualCNN(64, 64, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*64, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x



def inference(model, device, test_loader, criterion, epoch, wav):
    model.eval()
    with torch.no_grad():
        inputs, labels, input_lengths, label_lengths = _data 
        inputs, labels = inputs.to(device), labels.to(device)
        # output = model(inputs, input_lengths)  # (batch, time, n_class)
        output=model(inputs)
        output = F.log_softmax(output, dim=2)
        
        output = output.transpose(0, 1) # (time, batch, n_class)
        test_loss += loss.item() / len(test_loader)
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
        decoded_preds, decoded_targets = list(map(str.strip, decoded_preds)), list(map(str.strip, decoded_targets))
        return decoded_preds

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


device = 'cuda'

hparams = {
        "n_cnn_layers": 4,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 4,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": 4,
        "epochs": 60
    }



## model = Network()

model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)


model = model.to(device)
criterion = nn.CTCLoss(blank=0, reduction='mean').to(device)
epochs = 60

optimizer = optim.Adam(model.parameters(), hparams['learning_rate'])

model, optimizer, epoch_num = load_checkpoint(model, optimizer, "checkpoints/final_epoch_61.model")
tsvs = pd.read_csv("Data/PartB_Telugu/Train/Transcription_LT_Sequence_Frame_Level_200_actual.tsv",
    sep="\t", index_col=0, header =None)[1]

wav = input("Enter WAV file: ")
inp_id = wav.split(".")[0]
wav, sr = librosa.load("Data/PartB_Telugu/Train/Audio/{}".format(wav))
spectrograms = valid_audio_transforms(torch.Tensor(wav)).unsqueeze(0)


outputs = model(spectrograms.unsqueeze(0).to(device))
actual = tsvs[int(inp_id)].lower()
label = torch.Tensor(text_transform.text_to_int(actual.lower()))
text_transform = TextTransform()
outputs = F.log_softmax(outputs, dim=2)
outs = GreedyDecoder(outputs,
              label.reshape(1,-1),
              [len(label)])
print(outs)
print(wer(outs[0], outs[1]))
