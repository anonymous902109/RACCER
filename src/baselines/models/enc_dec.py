import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class EncoderDecoder:

    def __init__(self):
        self.enc = Encoder()
        self.dec = Decoder()

    def train(self, dataset):
        try:
            self.load('trained_models/gridworld')
            self.enc.eval()
            self.dec.eval()
            return
        except FileNotFoundError:
            self.loss = nn.MSELoss()
            self.optim = optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=0.0001)

            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size

            train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=256, shuffle=True)

            n_ep = 10000
            for i in range(n_ep):
                for x, y in train_loader:
                    self.enc.train()
                    self.dec.train()

                    enc = self.enc(x)
                    dec = self.dec(enc)

                    loss = self.loss(dec, y)

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                if i % 100 == 0:
                    # evaluation on test data
                    self.enc.eval()
                    self.dec.eval()
                    total_loss = 0.0

                    for x, y in test_loader:
                        test_enc = self.enc(x)
                        test_dec = self.dec(test_enc)

                        test_loss = self.loss(test_dec, y)

                        total_loss += test_loss.item()

                    print('Epoch = {}, Test loss = {}'.format(i, total_loss / len(test_data)))
                    self.save()

    def encode(self, x):
        self.enc.eval()
        self.dec.eval()
        return self.enc.float()(x.float()).detach()

    def decode(self, x):
        self.dec.eval()
        self.enc.eval()
        return self.dec.float()(x.float()).detach()

    def save(self):
        self.enc.save('gridworld_optim_enc.zip')
        self.dec.save('gridworld_optim_dec.zip')

    def load(self, path):
        checkpoint_enc = torch.load(path + '_enc.zip')
        self.enc.load_state_dict(checkpoint_enc)

        checkpoint_dec = torch.load(path + '_dec.zip')
        self.dec.load_state_dict(checkpoint_dec)

        self.enc.eval()
        self.dec.eval()

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        layers = []

        layers.append(nn.Linear(8, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 5))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    def save(self, path):
        torch.save(self.state_dict(), path)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        layers = []

        layers.append(nn.Linear(5, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 8))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

