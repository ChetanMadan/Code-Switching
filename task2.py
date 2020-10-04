#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import pandas as pd
import torch.optim as optim
import numpy as np
import csv
from data_load import TextTransform
from difflib import SequenceMatcher
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import librosa
from librosa.core import stft, magphase
from glob import glob
from torch import autograd
import csv
from data_load import CodeSwitchDataset
import zipfile


# In[2]:


def pad(wav, trans, lang):
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
        ext = wav[:diff]
        wav = np.append(wav, wav[:diff])
        ratio = int(len(trans)*diff/len(wav))
        trans +=trans[:ratio]
    return wav, trans

def preprocess(data):
    #print(data)
    inputs = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    for (wav, sr, trans, lang) in data:
        #wav, trans  = pad(wav, trans, lang)
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
    #spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    return inputs, labels, input_lengths, label_lengths


# In[3]:


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


# In[4]:


train_dataset = CodeSwitchDataset(lang = 'Telugu', mode = "train")
validation_split = .2
shuffle_dataset = True
random_seed= 42
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


# In[5]:


train_loader = DataLoader(train_dataset,
                          batch_size=4,
                          drop_last=True,
                          num_workers = 6,
                          sampler = train_sampler,
                         collate_fn = lambda x: preprocess(x))
test_loader = DataLoader(valid_dataset,
                          batch_size=4,
                          drop_last=True,
                          num_workers = 6,
                          sampler = valid_sampler,
                         collate_fn = lambda x: preprocess(x))


# In[6]:


device = torch.device('cuda')


# In[7]:


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim = 4, num_layers = 4):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional = True)
        #self.drop = nn.Dropout(0.25)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        #print(type(self.lstm))
        #print(x.shape)
        lstm_out, hidden = self.lstm(x)
        #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.linear(lstm_out)
        return out, hidden
    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.hidden= nn.Linear(32*12*12, 128)
        self.drop = nn.Dropout(0.5)
        self.out1 = nn.Linear(128, 64)
        self.out2 = nn.Linear(64, 2)
        self.act = nn.ReLU()
        self.bgru = nn.LSTM(input_size=32, hidden_size=64)

    def forward(self, x):
        x = self.act(self.conv(x)) # [batch_size, 4, 30, 30]
        x = self.act(self.conv2(x)) # [batch_size, 8, 28, 28]
        x = self.act(self.conv2(x)) # [batch_size, 16, 26, 26]
        x = self.act(self.conv2(x)) # [batch_size, 32, 24, 24]
        x = self.pool(x) # [batch_size, 32, 12, 12]
        x = self.drop(x)
        x = self.hidden(x) # [batch_size, 128]
        x = self.out1(x) # [batch_size, 64]
        x = self.bgru(x)
        x = self.bgru(x)
        x = self.out2(x) # [batch_size, 2]
        return x


# In[8]:


torch.cuda.empty_cache()


# In[9]:


def train(model, device, train_loader, criterion, optimizer, epoch, iter_meter, writer):
    model.train()
    data_len = len(train_loader.dataset)
    total_loss=0
    LR=0
    train_loss=0
    avg_acc=0
    acc=[]
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
        #print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

        output, _ = model(wav)
        output = F.log_softmax(output, dim=1)
        output = output.transpose(0,1)
        #print(output.shape)
        loss = criterion(output, labels, input_lengths, label_lengths)
        #print(loss)
        total_loss+=loss
        loss.backward()
        
        optimizer.step()
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
            
    print(decoded_preds[0])
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


# In[10]:


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

            output, _ = model(inputs)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=1)
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


# In[11]:


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


# In[12]:


LEARNING_ANNEAL = 1.01


# In[13]:


model = Model(input_dim=1025,
              hidden_dim=1024,
              batch_size=4,
              output_dim=4,
              num_layers=4)
#model.half()
model = model.to(device)
criterion = nn.CTCLoss(blank = 3).to(device)
epochs = 40
optimizer = optim.Adam(model.parameters(), 1e-3)


# In[ ]:


writer = SummaryWriter('train_logs/')
iter_meter = IterMeter()
#checkpoint = torch.load("checkpoints/ckpt_epoch_22_batch_id_822.pth")
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
model.to(device)
model.train()
for epoch in range(1, epochs+1):
    train(model, device, train_loader, criterion, optimizer, epoch, iter_meter, writer)
    test(model, device, test_loader, criterion, epoch, writer)
    
model.eval().cpu()
save_model_filename = "final_epoch_" + str(epoch + 1) + "_batch_id_" + str(batch_idx + 1) + ".model"
save_model_path = os.path.join("checkpoints", save_model_filename)
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_model_path)

print("\nDone, trained model saved at", save_model_path)

