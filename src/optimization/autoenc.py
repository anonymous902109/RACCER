from typing import List, Union
import torch.nn as nn
import numpy as np
import torch
import pandas as pd


class AutoEncoder(nn.Module):
    def __init__(self, layers: List):
        """
        Parameters
        ----------
        layers:
            List of layer sizes.
        """
        super(AutoEncoder, self).__init__()

        if len(layers) < 2:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        self._input_dim = layers[0]
        latent_dim = layers[-1]

        # Autoencoder components
        lst_encoder = []
        for i in range(1, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.ReLU())

        self.encoder = nn.Sequential(*lst_encoder)
        self.encoder = nn.Sequential(self.encoder, nn.Linear(layers[-2], latent_dim))

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder.append((nn.ReLU()))
        self.decoder = nn.Sequential(*lst_decoder)

        self.decoder = nn.Sequential(
            self.decoder,
            nn.Linear(layers[1], self._input_dim)
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    def encode(self, x):
        return self.encoder.float()(x.float())

    def decode(self, z):
        return self.decoder.float()(z.float())

    def forward(self, x):
        # split up the input in a mutable and immutable part
        x = x.clone()

        # the mutable part gets encoded
        z = self.encode(x)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x = recon

        return x

    def predict(self, data):
        return self.forward(data)

    def fit(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        xtest: Union[pd.DataFrame, np.ndarray],
        epochs=50,
        lr=1e-3,
        batch_size=64,
    ):
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values

        if isinstance(xtest, pd.DataFrame):
            xtest = xtest.values

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr
        )

        self.train()

        self.criterion = nn.MSELoss()

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        print("Start training of Variational Autoencoder...")

        for epoch in range(epochs):
            # Initialize the losses
            train_loss = 0
            train_loss_num = 0

            # Train for all the batches
            for data in train_loader:
                data = data.view(data.shape[0], -1)
                data = data.float()

                # forward pass
                reconstruction = self(data)

                loss = self.criterion(reconstruction, data)

                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()

                # Collect the ways
                train_loss += loss.item()
                train_loss_num += 1

            ELBO[epoch] = train_loss / train_loss_num

            self.evaluate(xtest, epoch, epochs)

        print("... finished training of Variational Autoencoder.")

        self.dataset = np.concatenate([xtrain, xtest])

    def evaluate(self, test_data, epoch, epochs, batch_size=64):
        self.eval()

        if isinstance(test_data, pd.DataFrame):
            test_data = test_data.values

        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )

        test_loss = 0.0
        i_batches = 0.0
        for data in test_loader:
            data = data.view(data.shape[0], -1)
            data = data.float()

            # forward pass
            reconstruction = self(data)

            loss = self.criterion(reconstruction, data)

            # Collect the ways
            test_loss += loss.item()
            i_batches += 1

        print(
            "[Epoch: {}/{}] [Test MSE: {:.6f}]".format(
                epoch, epochs, test_loss/i_batches
            )
        )

        self.encoder.train()

    def max_diff(self, x):
        self.eval()

        diffs = []

        enc_fact = self.encode(torch.tensor(x).squeeze())
        for i in self.dataset:
            i_tensor = torch.tensor(i).squeeze()
            enc_i = self.encode(i_tensor)

            diff = sum(abs(torch.subtract(enc_i, enc_fact))).item()

            diffs.append(diff)

        return max(diffs)
