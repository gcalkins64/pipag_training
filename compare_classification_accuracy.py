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
from plotting import plot_latent_space_with_clusters, seabornSettings  # type: ignore
import glob
import re
import joblib  # safer and more compact than pickle for sklearn models
warnings.simplefilter("ignore", FutureWarning)

def main():
    seabornSettings()
    sns.set_palette("Paired")
    # Set your directory path
    basePath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/"

    # Structure - run once for each input data, but can run multiple hyperparameters within each input file
    # INPUTS
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/UOP_inc_lit_disps_5000_data_energy_scaled_downsampled_.json"
    # modes = [0,1]
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250429_155508_5_4'
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250429_183447_5_5'

    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag/data/UOP_near_crash_5000_data_energy_scaled_downsampled_.json"
    # modes = [0,2]
    inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/UOP_near_crash_steeper_5000_data_energy_scaled_downsampled_.json"
    modes = [0,2]
    folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250429_155516_5_4'
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250506_190907_5_5'

    LDs = [5] #[4,5,6]
    NCs = [4] #[2,3,4,5,6]

    # load in json
    with open(inputDataPath, 'r') as f:
        inputData = json.load(f)

    Nsamples = len(inputData)
    # Load in samples, compute number with energy above zero at the end
    samples = np.array([inputData[f'sample{i}']['energy'] for i in range(Nsamples)])
    # Get all samples labels
    labels = np.array([inputData[f'sample{i}']['label'] for i in range(Nsamples)])
    # Get probabilities
    capture_prob, escape_prob, impact_prob = 0, 0, 0
    for ii in range(Nsamples):  # 0 capture, 1 escape, 2 impact
        if labels[ii] == 0:
            capture_prob += 1
        elif labels[ii] == 2:
            impact_prob += 1
        else:
            escape_prob += 1

    # Normalize capture and escape probabilities
    capture_prob /= Nsamples
    escape_prob /= Nsamples
    impact_prob /= Nsamples
    print("True Capture Prob: ", capture_prob)
    print("True Escape Prob: ", escape_prob)
    print("True Impact Prob: ", impact_prob)

    # TODO add something about which clusters are are looking at for the loops

    # LABEL 0 = CAPTURE
    # LABEL 1 = ESCAPE
    # LABEL 2 = IMPACT
    pred_capture_probs = np.zeros((len(LDs), len(NCs)))
    pred_failure_probs = np.zeros((len(LDs), len(NCs)))
    # Loop over all LDs and NCs
    for ll, LD in enumerate(LDs):
        for nn, NC in enumerate(NCs):
            # Load in encoder
            print(f"LD: {LD}, NC: {NC}")
            if len(LDs) > 1:
                pattern = rf"^gmvae_em_aerocapture_energy_(20250429|20250430)_\d{{6}}_{LD}_{NC}$"
                folder_path = [
                    f for f in os.listdir(basePath)
                    if os.path.isdir(os.path.join(basePath, f)) and re.fullmatch(pattern, f)
                ]
                if folder_path:
                    print(f"LD {LD}, NC {NC} → {folder_path}")
                    folder_path = os.path.join(basePath, folder_path[0])
                else:
                    print(f"LD {LD}, NC {NC} → no match")


            # Get the file string after "encoder"
            suffix = [file for file in os.listdir(folder_path) if file.startswith("encoder")][0][8:-3]

            # Load in decoder and params
            encoder, params, em_reg = loadEncoderAndParams(folder_path, suffix, data_dim=64, latent_dim=LD, hidden_dims=[32, 16])  # type: ignore

            # For each GMVAE, figure out which mixands describe which clusters as which data cluster has the smallest mahalanobis distance from each cluster mean / variance in latent space
            # run all samples through encoder
            assigned_cluster_inds = []
            for ii in range(NC):  # for each GMVAE cluster
                mahalanobis_distances = [[], []]  # Two lists - one for true cluster 0 (escape/crash) and one for true cluster 1 (capture)
                encoded_samples = []
                for kk in range(Nsamples): # Loop over all samples
                    # Compute mahalanobis distance from sample to cluster mean / variance
                    z, logsigmasq = encoder.forward(torch.tensor(samples[kk][np.newaxis, :]).float())
                    encoded_samples.append(z)
                    mu_c = params['mu_c'][ii].clone().detach()
                    logsigmasq_c = params['logsigmasq_c'][ii].clone().detach()
                    mahalanobis_distance = torch.sqrt((z - mu_c) @ torch.inverse(torch.diag(torch.exp(logsigmasq_c))) @ (z - mu_c).T)
                    assigned_label = labels[kk] if labels[kk] != 2 else 1
                    mahalanobis_distances[assigned_label].append(mahalanobis_distance.detach().numpy())
                # Get mean mahanalobis distance for this mixand to each true cluster
                mean_mahalanobis_distances = [np.mean(mahalanobis_distances[0]), np.mean(mahalanobis_distances[1])]
                print(f"Mean Mahalanobis distance for mixand {ii} to failure cluster: {mean_mahalanobis_distances[1]}")
                print(f"Mean Mahalanobis distance for mixand {ii} to capture cluster: {mean_mahalanobis_distances[0]}")
                # Assign mixand to cluster with smallest mean mahalanobis distance
                assigned_cluster = np.argmin(mean_mahalanobis_distances)
                assigned_cluster_inds.append(assigned_cluster)

            # Print out assigned cluster inds
            print(assigned_cluster_inds)

            # Plot latent space with samples and color all mixands by their assigned cluster to check
            cluster_labels, cluster_colors = [], []
            bad_num, good_num = 0, 0
            for aa, assigned_ind in enumerate(assigned_cluster_inds):
                if assigned_ind == 1:
                    if 2 in modes:
                        cluster_labels.append(f'Impact {bad_num}')
                    else:
                        cluster_labels.append(f'Escape {bad_num}')
                    cluster_colors.append('C2')
                    bad_num += 1
                else:  # assigned_ind == 0
                    cluster_labels.append(f'Capture {good_num}')
                    cluster_colors.append('C0')
                    good_num += 1

            if 2 in modes:
                labels = np.where(labels == 2, 1, labels)  # Change all impact labels to 1 for plottinh
            encoded_samples = np.squeeze(np.array([t.detach().numpy() for t in encoded_samples]))
            names = ['Capture', 'Escape'] if 1 in modes else ['Capture', 'Impact']
            plot_latent_space_with_clusters(encoded_samples, labels, NC, params['mu_c'], params['logsigmasq_c'], os.path.join(folder_path, f'predicted_latent_clusters_LD{LD}_NC{NC}'), names, ['C1', 'C3'], cluster_labels, cluster_colors, dpi=300, titleTag=f" LD: {LD}, NC: {NC}")
            plt.show()

            # compute true cluster probability by summing probability for all mixands in that cluster
            pred_capture_prob, pred_failure_prob = 0, 0
            for aa, assigned_ind in enumerate(assigned_cluster_inds):
                if assigned_ind == 0:
                    pred_capture_prob += params['pi_c'][aa].detach().numpy()
                else:
                    pred_failure_prob += params['pi_c'][aa].detach().numpy()
            pred_capture_probs[ll, nn] = pred_capture_prob
            pred_failure_probs[ll, nn] = pred_failure_prob

            # pass all samples through the encoder and perform em step to see which cluster they are assigned to
            pred_labels = []
            for ii in range(Nsamples):
                sample = torch.tensor(samples[ii][:,np.newaxis].T).float()
                z, logsigmasq = encoder.forward(sample)
                gamma_c, _, _ = em_step(z, z, logsigmasq, params, em_reg)  # type: ignore
                cluster_ind = np.argmax(gamma_c.detach().numpy())
                pred_labels.append(assigned_cluster_inds[cluster_ind])
            pred_labels = np.array(pred_labels)

            # Compute number of false assignments in each cluster
            # Find indices where labels and pred_labels are different
            false_assignments = np.where(labels != pred_labels)[0]
            print(f"Number of false assignments: {len(false_assignments)}")
            print(f"False assignment %: {len(false_assignments) / Nsamples * 100}")

            # Compute number of false assignments in each cluster
            false_assignments_capture = np.where((labels == 0) & (pred_labels != 0))[0]
            false_assignments_failure = np.where((labels == 1) & (pred_labels != 1))[0]
            print(f"Number of false assignments to capture cluster: {len(false_assignments_capture)}")
            print(f"Number of false assignments to failure cluster: {len(false_assignments_failure)}")
            # Get false assignment percentage
            false_assignment_capture_percentage = len(false_assignments_capture) / len(np.where(labels == 0)[0])
            false_assignment_failure_percentage = len(false_assignments_failure) / len(np.where(labels == 1)[0])
            print(f"False assignment percentage to capture cluster: {false_assignment_capture_percentage*100}")
            print(f"False assignment percentage to failure cluster: {false_assignment_failure_percentage*100}")

            # Plot input data energy colored by assigned cluster (pred_label)
            fig, ax = plt.subplots()
            for ii in range(Nsamples):
                if pred_labels[ii] == 0:  # Capture
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
            ax.plot([], color='C0', label='Correctly Predicted Capture')
            if 1 in modes:
                ax.plot([], color='C2', label='Correctly Predicted Escape')
            else:
                ax.plot([], color='C2', label='Correctly Predicted Impact')
            ax.plot([], color='C1', label='Incorrectly Predicted Capture')
            if 1 in modes:
                ax.plot([], color='C3', label='Incorrectly Predicted Escape')
            else:
                ax.plot([], color='C3', label='Incorrectly Predicted Impact')
            ax.axhline(0, color='black', linestyle='--')
            ax.legend(loc='lower left')
            plt.title(f"LD: {LD}, NC: {NC}")
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"predicted_clusters_LD{LD}_NC{NC}.png"), dpi=300)
            plt.close()


    if len(LDs) > 1:
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
            print(f"{LD} & " + " & ".join([f"{pred_failure_probs[ll, nn]:.4f}" for nn in range(len(NCs))]) + " \\\\")
        print("\\bottomrule")

        # TODO change these to percent error
        # Print capture probability error in latex table
        print("Capture Probability Percent Error")
        print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
        print("\\toprule")
        print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
        print("\\midrule")
        for ll, LD in enumerate(LDs):
            print(f"{LD} & " + " & ".join([f"{abs(pred_capture_probs[ll, nn] - capture_prob)/capture_prob*100:.4f}" for nn in range(len(NCs))]) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

        # Print escape probability error in latex table
        print("Escape Probability Percent Error")
        print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
        print("\\toprule")
        print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
        print("\\midrule")
        for ll, LD in enumerate(LDs):
            print(f"{LD} & " + " & ".join([f"{abs(pred_failure_probs[ll, nn] - escape_prob)/escape_prob*100:.4f}" for nn in range(len(NCs))]) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()


