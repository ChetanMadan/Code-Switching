import torch.nn as nn
import torch

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


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        n_out_classes = 4
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        #self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.hidden = nn.Linear(512, 128)
        self.drop = nn.Dropout(0.5)
        self.out1 = nn.Linear(128, 64)
        self.out2 = nn.Linear(128, n_out_classes)
        self.act = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 512))
        self.num_layers = 2
        self.batch_size = 4
        self.hidden_size = 128
        self.bgru1 = nn.GRU(input_size=64,
                            hidden_size=128,
                            num_layers=2,
                           batch_first=True)

    def forward(self, x, x_lengths):
        h0 = torch.rand(self.num_layers, self.batch_size, self.hidden_size).to(device)
        print("Initial: ", x.shape)
        x = self.act(self.conv(x))  # [batch_size, 4, 30, 30]
        print("conv: ", x.shape)
        x = self.act(self.conv1(x))  # [batch_size, 8, 28, 28]
        print("conv1: ", x.shape)
        x = self.act(self.conv2(x))  # [batch_size, 16, 26, 26]
        print("conv2: ", x.shape)
        x = self.act(self.conv3(x))  # [batch_size, 32, 24, 24]
        print("conv3: ", x.shape)
        #x = self.avg_pool(x)  # [batch_size, 32, 12, 12]
        #print("avg_pool: ", x.shape)
        #x = self.drop(x)
        x = x.permute(0,3,1,2)
        T = x.size(1)
        x = x.view(self.batch_size, T, -1)

        #x = self.hidden(x)  # [batch_size, 128]
        print("After reshape: ", x.shape)
        # x = self.out1(x) # [batch_size, 64]
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              x_lengths,
                                              enforce_sorted=False)
        
        x, h0 = self.bgru1(x, h0)
        print(x.shape)
        x, _ = nn.utils.rnn.pad_packed_sequence(x,
                                                batch_first=True)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        
        x = self.out2(x)  # [batch_size, 2]
        x = F.log_softmax(x, dim=1)
        x = x.view(batch_size, T, n_out_classes)

        
        #x = self.bgru2(x)
        #print(x.shape)
        #x = self.out2(x)  # [batch_size, 2]
        
        print(x.shape)
        return x













class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ### RNN ###
        self.rnn1 = nn.GRU(input_size=32,  # input is the output from CNN
                           hidden_size=hidden_size,
                           num_layers=1)

        self.rnn2 = nn.GRU(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=1)

        self.rnn3 = nn.GRU(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=1)

        self.rnn4 = nn.GRU(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=1)

        self.activation = nn.ReLU()

        ### END ###
        self.dense1 = nn.Linear(hidden_size, 3)

        ### CNN ###
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),    # reduce dimension of sequece by half
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),    # reduce dimension of sequece by half
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),    # reduce dimension of sequece by half
        )

    def forward(self, x, hidden, batch_size):
        x = self.conv1(x.double())  # inputs (1,3,batch_size)
        x = self.conv2(x.double())
        x = self.conv3(x.double())

        # Reshape batch for RNN training:
        x = x.reshape(batch_size, 1, 32)
        x, hidden = self.rnn1(x, hidden)  # inputs (seq_len,1,3)
        x = self.activation(x)
        x, hidden = self.rnn2(x, hidden)
        x = self.activation(x)
        x, hidden = self.rnn3(x, hidden)
        x = self.activation(x)
        x, hidden = self.rnn4(x, hidden)

        # x = x.select(0, maxlen-1).contiguous()

        x = x.view(-1, hidden_size)
        x = F.relu(self.dense1(x))
        return x, hidden  # Returns prediction for all batch_size timestamps. i.e [batch_size, 3]

    def init_hidden(self):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, 1, hidden_size).zero_())


def train():
    print("Training Initiated!")
    model.train()
    hidden = model.init_hidden()  # Initiate hidden
    for step, data in enumerate(train_set_all):
        X = data[0]  # Entire sequence
        y = data[1]  # [1,0,0] or [0,1,0] or [0,0,1]
        y = y.long()
        # print(y.size())
        ### Split sequence into batches:
        batch_size = 50  # split sequence into mini-sequences of size 50
        max_batches = int(X.size(2) / batch_size)

        for nbatch in range(max_batches):
            model.zero_grad()
            output, hidden = model(X[:, :, nbatch * batch_size:batch_size + nbatch * batch_size], Variable(hidden.data),
                                   batch_size)

            loss = criterion(output, torch.max(
                y[:, nbatch * batch_size:batch_size + nbatch * batch_size, :].reshape(batch_size, 3), 1)[1])
            loss.backward()
            optimizer.step()

        print(step)