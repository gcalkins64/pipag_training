import numpy as np
import os
import json
import sys
sys.path.append("/Users/gracecalkins/Local_Documents/local_code/pipag/pipag_base")
from gmvae_encoder import *  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal
import torch
import warnings
warnings.simplefilter("ignore", FutureWarning)

sns.set_style('whitegrid')
sns.set_palette("Set2")
sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,
                                'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, "font.family": "serif"})
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Set your directory path
basePath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/"

inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag/data/UOP_training_data/UOP_training_data_5000_scaled_downsampled_energy.json"
# load in json
with open(inputDataPath, 'r') as f:
    inputData = json.load(f)

Nsamples = 5000
# Get mean and variance of all input samples
samples = np.array([inputData[f'sample{i}']['energy'] for i in range(Nsamples)])
mean = np.mean(samples, axis=0)
variance = np.var(samples, axis=0)

LDs = [4,5,6]
NCs = [2,3,4,5,6]

for LD in LDs:
    for NC in NCs:
        print(f"Generating samples for LD: {LD}, NC: {NC}")
        # Get the folder in basePath that ends with LD{LD}_NC{NC}
        folder_path = os.path.join(basePath, [folder for folder in os.listdir(basePath) if folder.endswith(f"LD{LD}_NC{NC}")][0])

        # Get the file string after "encoder"
        suffix = [file for file in os.listdir(folder_path) if file.startswith("encoder")][0][8:-3]

        # Load in decoder and params
        decoder, params = loadDecoderAndParams(folder_path, suffix, data_dim=64, latent_dim=LD, hidden_dims=[16, 32, 64])  # type: ignore

        # Randomly sample Nsamples from a GMM with means / variances from params
        gen_samples = []
        for j in range(Nsamples):
            c = np.random.choice(NC, p=params['pi_c'].detach().numpy())
            mu_c = params['mu_c'][c].clone().detach()
            sigma_c = torch.exp(0.5 * params['logsigmasq_c'][c]).clone().detach()
            z = Normal(0, 1).sample(mu_c.shape) * sigma_c + mu_c
            mu_x = decoder.forward(z)[0].detach().numpy()
            gen_samples.append(mu_x)

        # Get mean and variance of generate samples
        gen_samples = np.array(gen_samples)
        gen_mean = np.mean(gen_samples, axis=0)
        gen_variance = np.var(gen_samples, axis=0)

        # Plot hair plot of generated samples with 3 sigma shading of input mean and variance and generated mean and variance
        plt.figure()
        for i in range(Nsamples):
            plt.plot(gen_samples[i], color='grey', alpha=0.1)
        plt.plot([], label='GMVAE Generated Samples', color='grey', alpha=0.1)
        plt.plot(mean, label='Input Mean', color='C0')
        plt.fill_between(range(len(mean)), mean - 3 * np.sqrt(variance), mean + 3 * np.sqrt(variance), alpha=0.5, label='Input 3 Sigma', color='C0')
        plt.plot(gen_mean, label='Generated Mean', color='C1')
        plt.fill_between(range(len(gen_mean)), gen_mean - 3 * np.sqrt(gen_variance), gen_mean + 3 * np.sqrt(gen_variance), alpha=0.5, label='Generated 3 Sigma', color='C1')
        plt.legend(loc = 'lower left')
        plt.xlabel('Downsample Index')
        plt.ylabel('Scaled Energy')
        plt.title(f"Latent Dim: {LD}, Clusters: {NC}")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f"generated_samples_LD{LD}_NC{NC}.png"), dpi=300)
        plt.close()


