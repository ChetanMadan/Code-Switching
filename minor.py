#!/usr/bin/env python
# coding: utf-8

"""
Take care:
    Data loader  - seems ok
    Batch first (Shape),
    Learning rate - seems ok
    Optimizer - seems ok
    Padding,
    Architecture,
    Transforms,
    Preprocessing - seems ok
    Decoder
"""
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
import torch
import torch.nn as nn
from librosa.core import stft, magphase
from torch import autograd
from data_load import CodeSwitchDataset, TextTransform


def pad(wav, trans, lang):
    """
    Never used, just to preserve the maximum length of languages
    """
    if lang == "Gujarati":
        max_len = 0
    elif lang == "Telugu":
        max_len = 529862
    elif lang == 'Tamil':
        max_len = 0
    else:
        raise Exception("Check Language")

    while len(wav) < max_len:
        diff = max_len - len(wav)
        wav = np.append(wav, wav[:diff])
        ratio = int(len(trans)*diff/len(wav))
        trans +=trans[:ratio]
    return wav, trans


def preprocess(data):
    """
    Never used, can be deleted (don't delete though)
    """
    inputs = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    for (wav, sr, trans, lang) in data:
        # wav, trans  = pad(wav, trans, lang)
        out = stft(wav, win_length=int(sr*0.02), hop_length=int(sr*0.01))
        out = np.transpose(out, axes=(1, 0))

        text_transform = TextTransform()
        trans = torch.Tensor(text_transform.text_to_int(trans.lower()))

        out = magphase(out)[0]
        out = torch.from_numpy(np.array([np.log(1 + x) for x in out]))
        inputs.append(out)
        labels.append(trans)
        input_lengths.append(out.shape[0])
        label_lengths.append(len(trans))
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    # spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    return inputs, labels, input_lengths, label_lengths


def preprocess_crnn(data, mode='train'):
    # print(data)
    inputs = []
    labels = []
    if mode == 'train':
        transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(n_mels=128, sample_rate = 22050, n_fft = 500, win_length=int(22050*0.02), hop_length=int(22050*0.01)),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
    else:
        transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(n_mels=128, sample_rate = 22050, n_fft = 500, win_length=int(22050*0.02), hop_length=int(22050*0.01)),
        )
    input_lengths = []
    label_lengths = []
    
    for (wav, sr, trans, lang) in data:
        #wav, trans  = pad(wav, trans, lang)
        out = transform(torch.Tensor(wav)).squeeze(0).transpose(0, 1)
        text_transform = TextTransform()
        trans = torch.Tensor(text_transform.text_to_int(trans.lower()))

        inputs.append(out)
        labels.append(trans)
        input_lengths.append(out.shape[0])
        label_lengths.append(len(trans))
    #inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    inputs =  nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1).transpose(2, 3)
    return inputs, labels, input_lengths, label_lengths


def GreedyDecoder(output, labels, label_lengths, blank_label=3, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    text_transform = TextTransform()
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



train_dataset = CodeSwitchDataset(lang='Telugu', mode="train")
validation_split = 0.2
shuffle_dataset = True
random_seed = 42
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset,
                          batch_size=4,
                          drop_last=True,
                          num_workers = 6,
                          sampler = train_sampler,
                          collate_fn = lambda x: preprocess_crnn(x, 'train'))

test_loader = DataLoader(train_dataset,
                          batch_size=4,
                          drop_last=True,
                          num_workers=6,
                          sampler=valid_sampler,
                         collate_fn=lambda x: preprocess_crnn(x, 'test'))


device = torch.device('cuda')
torch.cuda.empty_cache()
"""
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=4, num_layers=4):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=True)
        # self.drop = nn.Dropout(0.25)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # print(type(self.lstm))
        # print(x.shape)
        lstm_out, hidden = self.lstm(x)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.linear(lstm_out)
        return out, hidden
"""


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_out_classes = 4
        self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.hidden = nn.Linear(512, 128)
        self.drop = nn.Dropout(0.5)
        self.out1 = nn.Linear(8192, 512)
        self.out2 = nn.Linear(256, self.n_out_classes)
        self.act = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 512))
        self.num_layers = 2
        self.batch_size = 4
        self.hidden_size = 256
        self.bgru1 = nn.GRU(input_size=512,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)

    def forward(self, x, x_lengths):
        #print("batch_size, seq_len, _ = ", x.size())
        seq_len = x.size()[-1]
        h0 = torch.rand(self.num_layers, self.batch_size, self.hidden_size).to(device)
        #print("Initial: ", x.shape)
        # [batch_size, channels, height, width]
        x = self.act(self.conv(x))  # [batch_size, 4, feats, seq_len]
        x = self.act(self.conv1(x))  # [batch_size, 8, 28, 28]
        x = self.act(self.conv2(x))  # [batch_size, 16, 26, 26]
        x = self.act(self.conv3(x))  # [batch_size, 32, 24, 24]
        #x = self.avg_pool(x)  # [batch_size, 32, 12, 12]
        #print("avg_pool: ", x.shape)
        #x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        # [batch_size, width, channels, height]
        T = x.size(1)
        x = x.view(self.batch_size, T, -1)
        # [batch_size, width (time steps or length), channels * height]
        #x = self.hidden(x)  # [batch_size, 128]
        
        x = self.out1(x)
        # x = self.out1(x) # [batch_size, 64]
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              x_lengths,
                                              enforce_sorted=False,
                                              batch_first=True)
        x, h0 = self.bgru1(x, h0)
        x, _ = nn.utils.rnn.pad_packed_sequence(x,
                                                batch_first=True)
        x = x.contiguous() 
        x = x.view(-1, x.shape[2])
        
        x = self.out2(x)  # [batch_size, 2]
        x = F.log_softmax(x, dim=1)
        x = x.view(self.batch_size, seq_len, self.n_out_classes)
        # x = self.bgru2(x)
        # print(x.shape)
        # x = self.out2(x)  # [batch_size, 2]
        return x


# In[21]:


def train(model, device, train_loader, criterion, optimizer, epoch, iter_meter, writer):
    model.train()
    data_len = len(train_loader.dataset)
    total_loss=0
    LR = 0
    train_loss = 0
    avg_acc = 0
    acc = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (_data) in pbar:
        #bi, wav, label = batch_idx, wav, label
        for g in optimizer.param_groups:
            LR=g['lr']
        wav, labels, input_lengths, label_lengths = _data
        wav = wav.to(device)
        wav = wav.float()
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(wav, input_lengths)   #(batch, time, n_class) [4, 911, 3]
        output = output.transpose(0,1)
        #print(output.shape, labels.shape, len(input_lengths), len(label_lengths))
        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        #print(loss)
        total_loss+=loss

        iter_meter.step()
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
        decoded_preds, decoded_targets = list(map(str.strip, decoded_preds)), list(map(str.strip, decoded_targets))
        for j in range(len(decoded_preds)):
            s = SequenceMatcher(None, decoded_targets[j], decoded_preds[j])
            acc.append(s.ratio())

        avg_acc = sum(acc)/len(acc)
        writer.add_scalar("train_accuracy", avg_acc, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('CTCLoss', loss, epoch*len(train_loader)+1)
        writer.add_scalar('TLoss', total_loss, epoch*len(train_loader)+1)
        writer.add_scalar("Learning Rate", LR, epoch)
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(wav), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))
            print("Train Accuracy: {}, Train loss: {}".format(avg_acc, train_loss))
    for g in optimizer.param_groups:
        g['lr'] = g['lr']/LEARNING_ANNEAL
            
    #print(decoded_preds[0])
    if (epoch+1)%2 == 0:
        model.eval().cpu()
        ckpt_model_filename = "ckpt_epoch_" + str(epoch+1) + "_batch_id_" + str(batch_idx+1) + ".pth"
        ckpt_model_path = os.path.join("checkpoints/", ckpt_model_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_model_path)
        model.to(device).train()


def test(model, device, test_loader, criterion, epoch, writer):
    model.eval()
    training_loss, train_acc = 0, 0
    eer, total_eer = 0, 0
    test_loss=0
    acc = []
    with torch.no_grad():
        for batch_idx, _data in enumerate(test_loader):
            inputs, labels, input_lengths, label_lengths = _data 
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs, input_lengths)  # (batch, time, n_class)
            output = output.transpose(0, 1) # (time, batch, n_class)
            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)
            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            decoded_preds, decoded_targets = list(map(str.strip, decoded_preds)), list(map(str.strip, decoded_targets))
            for j in range(len(decoded_preds)):
                s = SequenceMatcher(None, decoded_targets[j], decoded_preds[j])
                acc.append(s.ratio())

            avg_acc = sum(acc)/len(acc)
            writer.add_scalar("test_accuracy", avg_acc, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            print("Test Accuracy: {}, Test loss: {}".format(avg_acc, test_loss))
        
    print(decoded_targets)
    print(decoded_preds)


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


# In[24]:


LEARNING_ANNEAL = 1.01


# In[26]:


"""
model = Model(input_dim=1025,
              hidden_dim=1024,
              batch_size=4,
              output_dim=4,
              num_layers=4)
#model.half()
"""
model = Network()
model = model.to(device)
criterion = nn.CTCLoss(blank=3, reduction='mean').to(device)
epochs = 50
optimizer = optim.Adam(model.parameters(), 1e-3)


writer = SummaryWriter('train_logs/')
iter_meter = IterMeter()
# checkpoint = torch.load("checkpoints/ckpt_epoch_22_batch_id_822.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
model.to(device)
model.train()

for epoch in range(1, epochs+1):
    train(model, device, train_loader, criterion, optimizer, epoch, iter_meter, writer)
    test(model, device, test_loader, criterion, epoch, writer)

model.eval().cpu()
save_model_filename = "final_epoch_" + str(epoch + 1)  + ".model"
save_model_path = os.path.join("checkpoints", save_model_filename)
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_model_path)

print("\nDone, trained model saved at", save_model_path)

