import numpy as np
import warnings
import json
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append("/Users/gracecalkins/Local_Documents/local_code/pipag/pipag_base")
import torch
from gmvae_encoder import *  # type: ignore
from plotting import plot_latent_space_with_clusters  # type: ignore
warnings.simplefilter("ignore", FutureWarning)

sns.set_style('whitegrid')
sns.set_palette("Paired")
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
# Load in samples, compute number with energy above zero at the end
samples = np.array([inputData[f'sample{i}']['energy'] for i in range(Nsamples)])
# Get all samples labels
labels = np.array([inputData[f'sample{i}']['label'] for i in range(Nsamples)])
# Get number of labels that are 1
capture_prob = np.sum(labels) / Nsamples
print("True Capture Prob: ", capture_prob)
print("True Escape Prob: ", 1 - capture_prob)

LDs = [4,5,6]
NCs = [2,3,4,5,6]

# LABEL 1 = CAPTURE
# LABEL 0 = ESCAPE
pred_capture_probs = np.zeros((len(LDs), len(NCs)))
pred_escape_probs = np.zeros((len(LDs), len(NCs)))
for ll, LD in enumerate(LDs):
    for nn, NC in enumerate(NCs):
        # Load in encoder
        print(f"LD: {LD}, NC: {NC}")
        # Get the folder in basePath that ends with LD{LD}_NC{NC}
        folder_path = os.path.join(basePath,
                                   [folder for folder in os.listdir(basePath) if folder.endswith(f"LD{LD}_NC{NC}")][0])

        # Get the file string after "encoder"
        suffix = [file for file in os.listdir(folder_path) if file.startswith("encoder")][0][8:-3]

        # Load in decoder and params
        encoder, params, em_reg = loadEncoderAndParams(folder_path, suffix, data_dim=64, latent_dim=LD, hidden_dims=[64, 32, 16])  # type: ignore

        # For each GMVAE, figure out which mixands describe which clusters as which data cluster has the smallest mahalanobis distance from each cluster mean / variance in latent space
        # run all samples through encoder
        assigned_cluster_inds = []
        for ii in range(NC):  # for each GMVAE cluster
            mahalanobis_distances = [[], []]  # Two lists - one for true cluster 0 (escape) and one for true cluster 1 (capture)
            encoded_samples = []
            for kk in range(Nsamples): # Loop over all samples
                # Compute mahalanobis distance from sample to cluster mean / variance
                z, logsigmasq = encoder.forward(torch.tensor(samples[kk][np.newaxis, :]).float())
                encoded_samples.append(z)
                mu_c = params['mu_c'][ii].clone().detach()
                logsigmasq_c = params['logsigmasq_c'][ii].clone().detach()
                mahalanobis_distance = torch.sqrt((z - mu_c) @ torch.inverse(torch.diag(torch.exp(logsigmasq_c))) @ (z - mu_c).T)
                mahalanobis_distances[labels[kk]].append(mahalanobis_distance.detach().numpy())
            # Get mean mahanalobis distance for this mixand to each true cluster
            mean_mahalanobis_distances = [np.mean(mahalanobis_distances[0]), np.mean(mahalanobis_distances[1])]
            print(f"Mean Mahalanobis distance for mixand {ii} to escape cluster: {mean_mahalanobis_distances[0]}")
            print(f"Mean Mahalanobis distance for mixand {ii} to capture cluster: {mean_mahalanobis_distances[1]}")
            # Assign mixand to cluster with smallest mean mahalanobis distance
            assigned_cluster = np.argmin(mean_mahalanobis_distances)
            assigned_cluster_inds.append(assigned_cluster)

        # Plot latent space with samples and color all mixands by their assigned cluster to check
        cluster_labels, cluster_colors = [], []
        for aa, assigned_ind in enumerate(assigned_cluster_inds):
            if assigned_ind == 1:
                cluster_labels.append(f'Capture {aa}')
                cluster_colors.append('C2')
            else:  # assigned_ind == 0
                cluster_labels.append(f'Escape {aa}')
                cluster_colors.append('C0')
        encoded_samples = np.squeeze(np.array([t.detach().numpy() for t in encoded_samples]))
        plot_latent_space_with_clusters(encoded_samples, labels, NC, params['mu_c'], params['logsigmasq_c'], os.path.join(folder_path, f'predicted_latent_clusters_LD{LD}_NC{NC}'), ['Escape', 'Capture'], ['C1', 'C3'],cluster_labels, cluster_colors, dpi=300, titleTag=f" LD: {LD}, NC: {NC}")

        # compute true cluster probability by summing probability for all mixands in that cluster
        pred_capture_prob, pred_escape_prob = 0, 0
        for aa, assigned_ind in enumerate(assigned_cluster_inds):
            if assigned_ind == 0:
                pred_escape_prob += params['pi_c'][aa].detach().numpy()
            else:
                pred_capture_prob += params['pi_c'][aa].detach().numpy()
        pred_capture_probs[ll, nn] = pred_capture_prob
        pred_escape_probs[ll, nn] = pred_escape_prob

        # pass all samples through the encoder and perform em step to see which cluster they are assigned to
        pred_labels = []
        for ii in range(Nsamples):
            sample = torch.tensor(samples[ii][:,np.newaxis].T).float()
            z, logsigmasq = encoder.forward(sample)
            gamma_c, _, _ = em_step(z, z, logsigmasq, params, em_reg)  # type: ignore
            cluster_ind = np.argmax(gamma_c.detach().numpy())
            pred_labels.append(assigned_cluster_inds[cluster_ind])

        # Compute number of false assignments in each cluster
        # Find indices where labels and pred_labels are different
        false_assignments = np.where(labels != pred_labels)[0]
        print(f"Number of false assignments: {len(false_assignments)}")

        # Plot input data energy colored by assigned cluster (pred_label)
        fig, ax = plt.subplots()
        for ii in range(Nsamples):
            if pred_labels[ii] == 0:  # Escape
                if pred_labels[ii] != labels[ii]:
                    ax.plot(samples[ii], color='C1', alpha=0.5)
                else:
                    ax.plot(samples[ii], color='C0', alpha=0.5)
            else:
                if pred_labels[ii] != labels[ii]:
                    ax.plot(samples[ii], color='C3', alpha=0.5)
                else:
                    ax.plot(samples[ii], color='C2', alpha=0.5)
        ax.set_ylabel('Scaled Energy')
        ax.set_xlabel('Downsample Index')
        ax.plot([], color='C0', label='Correctly Predicted Escape')
        ax.plot([], color='C2', label='Correctly Predicted Capture')
        ax.plot([], color='C1', label='Incorrectly Predicted Escape')
        ax.plot([], color='C3', label='Incorrectly Predicted Capture')
        ax.axhline(0, color='black', linestyle='--')
        ax.legend(loc='lower left')
        plt.title(f"LD: {LD}, NC: {NC}")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f"predicted_clusters_LD{LD}_NC{NC}.png"), dpi=300)
        plt.close()


# Print a booktabs latex table of the predicted capture probability for number of cluster and latent dimension
print("Predicted Capture Probabilities")
print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
print("\\toprule")
print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
print("\\midrule")
for ll, LD in enumerate(LDs):
    print(f"{LD} & " + " & ".join([f"{pred_capture_probs[ll, nn]:.4f}" for nn in range(len(NCs))]) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# Print a booktabs latex table of the predicted escape probability for number of cluster and latent dimension
print("Predicted Escape Probabilities")
print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
print("\\toprule")
print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
print("\\midrule")
for ll, LD in enumerate(LDs):
    print(f"{LD} & " + " & ".join([f"{pred_escape_probs[ll, nn]:.4f}" for nn in range(len(NCs))]) + " \\\\")
print("\\bottomrule")

# Print capture probability error in latex table
print("Capture Probability Error")
print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
print("\\toprule")
print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
print("\\midrule")
for ll, LD in enumerate(LDs):
    print(f"{LD} & " + " & ".join([f"{pred_capture_probs[ll, nn] - capture_prob:.4f}" for nn in range(len(NCs))]) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# Print escape probability error in latex table
print("Escape Probability Error")
print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
print("\\toprule")
print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
print("\\midrule")
for ll, LD in enumerate(LDs):
    print(f"{LD} & " + " & ".join([f"{pred_escape_probs[ll, nn] - (1 - capture_prob):.4f}" for nn in range(len(NCs))]) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")


