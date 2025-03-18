# -*- coding: utf-8 -*-
"""VAE-demo.ipynb

author: @gracecalkins
date: 2024-05-02
time: 16:30:00
"""


from typing import List, Any, Tuple
from torch import nn, Tensor
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json
import random
import time

def seabornSettings():
    sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,
                                    'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, })
    sns.set_style('whitegrid')
    # sns.set_palette('bright')
    sns.set_palette("Set2")
    return

class VAE(nn.Module):

    def __init__(self,
                 beta: float = 1.0,
                 latent_dim: int = 4,
                 input_dim: int = 64,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.beta = beta
        self.latent_dim = latent_dim 
        self.input_dim = input_dim

        # Build Encoder
        self.elayer1 = nn.Linear(self.input_dim, 64)
        self.elayer2 = nn.Linear(64, 32)
        self.elayer3 = nn.Linear(32, 16)
        self.fc_mu = nn.Linear(16, self.latent_dim)
        self.fc_var = nn.Linear(16, self.latent_dim)

        # Build Decoder
        self.dlayer1 = nn.Linear(self.latent_dim, 16)
        self.dlayer2 = nn.Linear(16, 32)
        self.dlayer3 = nn.Linear(32, 64)
        self.dlayer4 = nn.Linear(64, self.input_dim)

        # Activation Function
        self.act = nn.Tanh()
        self.actOuter = nn.ReLU()

    def encode(self, x: Tensor) -> List[Tensor]:
        x = self.elayer1(x)
        x = self.act(x)
        x = self.elayer2(x)
        x = self.act(x)
        x = self.elayer3(x)
        # x = self.act(x)

        # Generate latent mean and covariance
        mu = self.fc_mu(x)
        log_var = self.fc_var(x) # the output of fc_var is log covaraince

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        z = self.dlayer1(z)
        z = self.act(z)
        z = self.dlayer2(z)
        z = self.act(z)
        z = self.dlayer3(z)
        z = self.act(z)
        z = self.dlayer4(z)
        # z = self.actOuter(z)
        return z

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), x, mu, log_var]

    def loss_function(self,
                      data: Tensor
                      ) -> dict:
        mu,log_var = self.encode(data)
        # Build loss function
        data_hat = self.forward(data)[0]
        recons_loss = ((data - data_hat)**2).sum()
        kld_loss = torch.sum(-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)))

        loss = self.beta * recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)

        samples = self.decode(z)
        return samples

    def train(self,
              trn_data: Tensor,
              batch_size: int = 64,
              epochs: int = 1000):
        trn_dataloader = DataLoader(trn_data, batch_size=batch_size)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas= (0.9, 0.99))
        losses = []
        with tqdm(range(epochs), unit=' Epoch') as tepoch:
            for epoch in tepoch:
                epoch_loss = 0
                for batch_index, data in enumerate(trn_dataloader):
                    opt.zero_grad()
                    trn_loss = self.loss_function(data)['loss']
                    trn_loss.backward()
                    epoch_loss += trn_loss.item()
                    opt.step()

                    epoch_loss += trn_loss
                epoch_loss /= len(trn_dataloader)
                losses.append(epoch_loss.detach().item())
                tepoch.set_postfix(loss=epoch_loss.detach().numpy())

        return losses
    
def main():
    """Load in Data"""
    tag = 'test_vae'
    figPath = "figs/test_vae"
    resultsPath = os.path.join('data', 'test_vae')
    os.makedirs(figPath, exist_ok=True)
    os.makedirs(resultsPath, exist_ok=True)

    print("Loading Data...")
    ti = time.time()
    fileName = os.path.join('.', 'data', 'neptune_GMVAE_training_5000_data_scaled_downsampled.json')
    with open(fileName, 'r') as f:
        outputs = json.load(f)
    print(f'...Data Loaded in {time.time()-ti} seconds.')

    # SETTINGS
    beta = 1e-8
    latent_dim = 4
    batch_size = 2500
    epochs = 1_000

    Nsamples = 3274
    random.seed(0)
    ### Filter indices with label == 1
    sample_numbers_with_label_1 = [
        int(key.replace("sample", "")) for key, value in outputs.items() if value["label"] == 1
    ]

    ### Ensure there are at least Nsamples samples
    if len(sample_numbers_with_label_1) < Nsamples:
        raise ValueError("Not enough samples with label == 1.")
    ### Randomly select 2500 indices
    selected_samples = random.sample(sample_numbers_with_label_1, Nsamples)

    # for each of these samples, get the data at the selected indices
    # Downsample the data at 64 times
    data = np.zeros((Nsamples, 64))
    for jj, sample in enumerate(tqdm(selected_samples)):
        vels = np.array(outputs[f'sample{jj}']['vel'])[:]
        data[jj,:] = vels[:]

    print(f'Using {data.shape[0]} samples with {data.shape[1]} features.')
    seabornSettings()

    """After buiding the VAE, we start training the model using given data."""

    # Transform data into torch tensors
    trn_data = torch.from_numpy(data).type(torch.FloatTensor)
    # Claim a new VAE object
    vae = VAE(beta=beta, latent_dim=latent_dim)
    # Start training
    losses = vae.train(trn_data, epochs=epochs, batch_size=batch_size)

    # Plot the input data
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for i in trange(Nsamples):
        ax.scatter(np.arange(0, len(data[i]), 1), data[i], alpha=0.3, color='C0')
    plt.ylabel("Scaled Velocity")
    plt.xlabel("Sample Index Number")
    plt.title("Input Data")
    plt.savefig(os.path.join(figPath, tag+'_input_data.png'), dpi=300)
    # plt.show()

    # Visualize the epoch losses
    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.title(f"Training Loss, beta = {beta}, latent dim = {latent_dim}, batch size = {batch_size}")
    plt.savefig(os.path.join(figPath, tag+f'_{beta}_{latent_dim}_{batch_size}_{epochs}_training_loss.png'))
    # plt.show()

    """The next step is to compare the generated results with true samples."""

    gen_data = vae.sample(Nsamples).detach().numpy()
    # plot generative results
    fig, axs = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    for i in trange(Nsamples):
        axs[0].plot(data[i], alpha=0.3)
        axs[1].plot(gen_data[i], alpha=0.3)
    axs[0].set_title('True Data')
    axs[1].set_title(f'Synthesized Data, beta = {beta}, latent dim = {latent_dim}, batch size = {batch_size}')
    plt.savefig(os.path.join(figPath, tag+f'_{beta}_{latent_dim}_{batch_size}_{epochs}_synthesized_data.png'),dpi=300)
    # plt.show()
    print(f'DONE, Figures saved in {figPath}')

if __name__ == "__main__":
    main()