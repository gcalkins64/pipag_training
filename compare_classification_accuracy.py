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

def plot_table_heatmap(table, LDs, NCs, title, save_path=None, cmap="viridis", percent=False):
    """
    Plot a heatmap of a 2D table indexed by LDs (rows) and NCs (columns).

    table      : 2D numpy array (len(LDs) x len(NCs))
    LDs        : list of latent dimensions (row labels)
    NCs        : list of cluster counts (column labels)
    title      : title of the plot
    save_path  : if provided, saves figure to this path
    cmap       : matplotlib colormap
    percent    : if True, formats annotations as percentages
    """
    fig, ax = plt.subplots(figsize=(1*len(NCs)+1, 1.0*len(LDs)+1))

    if percent:
        annot_data = np.round(table * 100, 2)
        fmt = ".2f"
    else:
        annot_data = np.round(table, 4)
        fmt = ".4f"

    sns.heatmap(
        table if not percent else table * 100,
        annot=annot_data,
        fmt=fmt,
        cmap=cmap,
        xticklabels=NCs,
        yticklabels=LDs,
        cbar_kws={'label': 'Percent (%)' if percent else 'Value'},
        ax=ax
    )

    ax.set_xlabel("Number of Clusters (NC)")
    ax.set_ylabel("Latent Dimension (LD)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    # plt.show()
    plt.close(fig)

def main():
    seabornSettings()
    sns.set_palette("Paired")
    # Set your directory path
    basePath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/"

    # Structure - run once for each input data, but can run multiple hyperparameters within each input file
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/UOP_uniform_pGRAM_2000_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/UOP_near_crash_steeper_near_escape_COMBINED_5000_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/UOP_inc_lit_disps_5000_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/UOP_near_crash_steeper_5000_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/UOP_poly_truth_1500_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/1_near_escape_fnpag_2000_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/1_near_crash_fnpag_2000_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/1_near_escape_new_2000_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/1_near_crash_new_2000_data_energy_scaled_downsampled_.json"
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/1_uniform_new_2000_data_energy_scaled_downsampled_.json"

    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gumixture_1000_data_velocity_fpa_scaled_downsampled_.json"  # gu mixture training data dp = 1.5
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gaussian_dp2_1000_data_velocity_fpa_scaled_downsampled_.json"  # Gaussian dp = 2
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_studentsT_dp15_1000_data_velocity_fpa_scaled_downsampled_.json"  # students T, dp = 1.75, DOF = 4
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gumixture_1000_data_energy_scaled_downsampled_.json"  # ENERGY, gu mixture training data dp = 1.5
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gumixture_dp15_more_downwards_1000_data_energy_scaled_downsampled_.json"  # Energy, gu mixture, dp = 1.5, FPA biased downwards to get more crashes
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gaussian_dp15_1000_data_energy_scaled_downsampled_.json"  # Energy, gaussian, DP = 1.5
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gaussian_dp16_1000_data_energy_scaled_downsampled_.json"  # Energy, gaussian, DP = 1.6
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gaussian_dp17_1000_data_energy_scaled_downsampled_.json"  # Energy, gaussian, DP = 1.7
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gaussian_dp18_1000_data_energy_scaled_downsampled_.json"  # Energy, gaussian, DP = 1.8
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gaussian_dp19_1000_data_energy_scaled_downsampled_.json"  # Energy, gaussian, DP = 1.9
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_gaussian_dp2_1000_data_energy_scaled_downsampled_.json"  # Energy, gaussian, DP = 2
    # inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_studentsT_dp175_1000_data_energy_scaled_downsampled_.json"  # dp = 1.75 students T, DOF = 4
    inputDataPath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/rev_studentsT_dp15_1000_data_energy_scaled_downsampled_.json"  # dp = 1.5 students T, DOF = 4

    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250514_182106_5_5'  # Combined data
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250512_200948_5_5'  # Uniform data
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250515_204903_6_7'  # Uniform data
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250515_205329_6_8'  # Uniform data
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250516_183650_6_6'  # Combined data, larger arch
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250516_183701_6_7'  # Combined data, larger arch
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250522_164110_6_6'  # Polynomial uniform truth, larger arch
    # folder_path =  '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250429_155516_5_4'  # Crash
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250429_183447_5_5'  # Escape
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_near_crash_new_20250601_065902_L5_C5_retrained'  # near crash retrained
    # folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_near_escape_new_20250601_073224_L7_C3_retrained'  # near escape retrained

    saveTag = 'rev_gumixture_energy_biased_down_eval_studentst_15'  # tag to add to saved files
    saveLoopFigs = False  # Whether to save the figures within the loop

    LDs = [3,4,5,6,7,8,9]  #[4,5,6]
    NCs = [3,4,5,6,7,8,9]  #[2,3,4,5,6]
    # hds = [30,20,10]
    # hds = [24,18,12]
    hds = [36,24,12]

    # LDs = [5,6,7]  # Latent dimensions to test
    # NCs = [2,3,4,5,6]  # Number of clusters to test

    figPath = os.path.join("/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs/", f"gmvae_comparison_{saveTag}_{hds[0]}")
    if not os.path.exists(figPath):
        os.makedirs(figPath)

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

    # LABEL 0 = CAPTURE
    # LABEL 1 = ESCAPE
    # LABEL 2 = IMPACT
    pred_capture_probs = np.zeros((len(LDs), len(NCs)))
    pred_escape_probs = np.zeros((len(LDs), len(NCs)))
    pred_crash_probs = np.zeros((len(LDs), len(NCs)))

    false_capture_percent = np.zeros((len(LDs), len(NCs)))
    false_escape_percent = np.zeros((len(LDs), len(NCs)))
    false_crash_percent = np.zeros((len(LDs), len(NCs)))
    # Loop over all LDs and NCs
    for ll, LD in enumerate(LDs):
        for nn, NC in enumerate(NCs):
            # Load in encoder
            print(f"LD: {LD}, NC: {NC}")
            if len(NCs) > 1 or len(LDs) > 1:
                # pattern = rf"^gmvae_em_aerocapture_energy_(20250429|20250430)_\d{{6}}_{LD}_{NC}$"
                # pattern = rf"^gmvae_near_escape_(20250527|20250528)_\d{{6}}_L{LD}_C{NC}$"
                # pattern = rf"^gmvae_near_crash_(20250528|20250529)_\d{{6}}_L{LD}_C{NC}$"
                # pattern = rf"^gmvae_near_escape_new_20250601_\d{{6}}_L{LD}_C{NC}$"
                # pattern = rf"^gmvae_near_crash_new_20250601_\d{{6}}_L{LD}_C{NC}$"
                # pattern = rf"^gmvae_uniform_new_20250601_\d{{6}}_L{LD}_C{NC}$"
                # pattern = rf"^gmvae_rev_gu_mixture_(20260216|20260217)_\d{{6}}_L{LD}_C{NC}$"
                # pattern = rf"^gmvae_rev_gu_mixture_(20260216|20260217)_\d{{6}}_L{LD}_C{NC}_24_18_12$"
                pattern = rf"^gmvae_rev_gu_mixture_energy_biased_down_\d{{8}}_\d{{6}}_L{LD}_C{NC}_{hds[0]}_{hds[1]}_{hds[2]}$"
                folder_path = [
                    f for f in os.listdir(basePath)
                    if os.path.isdir(os.path.join(basePath, f)) and re.fullmatch(pattern, f)
                ]
                if folder_path:
                    print(f"🍀LD {LD}, NC {NC} → {folder_path}")
                    folder_path = os.path.join(basePath, folder_path[0])
                else:
                    print(f"🚨LD {LD}, NC {NC} → no match")
                    continue


            # Get the file string after "encoder"
            suffix = [file for file in os.listdir(folder_path) if file.startswith("encoder")][0][8:-3]
            print(suffix)

            # Load in decoder and params
            encoder, params, em_reg = loadEncoderAndParams(folder_path, suffix, data_dim=36, latent_dim=LD, hidden_dims=hds, oldFlag=False)  # type: ignore

            # For each GMVAE, figure out which mixands describe which clusters as which data cluster has the smallest mahalanobis distance from each cluster mean / variance in latent space
            # run all samples through encoder
            assigned_cluster_inds = []
            for ii in range(NC):  # for each GMVAE cluster
                mahalanobis_distances = [[], [], []]  # Two lists - one for each mode
                encoded_samples = []
                for kk in range(Nsamples): # Loop over all samples
                    # Compute mahalanobis distance from sample to cluster mean / variance
                    z, logsigmasq = encoder.forward(torch.tensor(samples[kk][np.newaxis, :]).float())
                    encoded_samples.append(z)
                    mu_c = params['mu_c'][ii].clone().detach()
                    logsigmasq_c = params['logsigmasq_c'][ii].clone().detach()
                    mahalanobis_distance = torch.sqrt((z - mu_c) @ torch.inverse(torch.diag(torch.exp(logsigmasq_c))) @ (z - mu_c).T)
                    assigned_label = labels[kk]
                    mahalanobis_distances[assigned_label].append(mahalanobis_distance.detach().numpy())
                # Get mean mahanalobis distance for this mixand to each true cluster
                mean_mahalanobis_distances = [np.mean(mahalanobis_distances[0]), np.mean(mahalanobis_distances[1]), np.mean(mahalanobis_distances[2])]
                for qq in range(3):
                    print(f"Mean Mahalanobis distance for mixand {ii} to true cluster {qq}: {mean_mahalanobis_distances[qq]}")
                # Assign mixand to cluster with smallest mean mahalanobis distance
                assigned_cluster = np.nanargmin(mean_mahalanobis_distances)
                assigned_cluster_inds.append(assigned_cluster)

            # Print out assigned cluster inds
            print(assigned_cluster_inds)

            # Plot latent space with samples and color all mixands by their assigned cluster to check
            cluster_labels, cluster_colors = [], []
            bad_num, bad_num1, good_num = 0, 0, 0
            for aa, assigned_ind in enumerate(assigned_cluster_inds):
                if assigned_ind == 1:
                    cluster_labels.append(f'Escape {bad_num}')
                    cluster_colors.append('C2')
                    bad_num += 1
                elif assigned_ind == 2:
                    cluster_labels.append(f'Impact {bad_num1}')
                    cluster_colors.append('C4')
                    bad_num1 += 1
                else:  # capture
                    cluster_labels.append(f'Capture {good_num}')
                    cluster_colors.append('C0')
                    good_num += 1

            encoded_samples = np.squeeze(np.array([t.detach().numpy() for t in encoded_samples]))
            names = ['Capture', 'Escape', 'Impact']
            if saveLoopFigs:
                plot_latent_space_with_clusters(encoded_samples, labels, NC, params['mu_c'], params['logsigmasq_c'], os.path.join(folder_path, f'predicted_latent_clusters_{saveTag}_LD{LD}_NC{NC}'), names, ['C1', 'C3', 'C5'], cluster_labels, cluster_colors, dpi=300, titleTag=f" LD: {LD}, NC: {NC}", legendFlag=False, figSize=(4,4))
            # plt.show()

            # compute true cluster probability by summing probability for all mixands in that cluster
            pred_capture_prob, pred_escape_prob, pred_crash_prob = 0, 0, 0
            for aa, assigned_ind in enumerate(assigned_cluster_inds):
                if assigned_ind == 0:
                    pred_capture_prob += params['pi_c'][aa].detach().numpy()
                elif assigned_ind == 1:
                    pred_escape_prob += params['pi_c'][aa].detach().numpy()
                else:
                    pred_crash_prob += params['pi_c'][aa].detach().numpy()
            pred_capture_probs[ll, nn] = pred_capture_prob
            pred_escape_probs[ll, nn] = pred_escape_prob
            pred_crash_probs[ll, nn] = pred_crash_prob

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
            false_assignments_escape = np.where((labels == 1) & (pred_labels != 1))[0]
            false_assignments_crash = np.where((labels == 2) & (pred_labels != 2))[0]
            print(f"Number of false assignments to capture cluster: {len(false_assignments_capture)}")
            print(f"Number of false assignments to escape cluster: {len(false_assignments_escape)}")
            print(f"Number of false assignments to impact cluster: {len(false_assignments_crash)}")
            # Get false assignment percentage
            false_assignment_capture_percentage = len(false_assignments_capture) / len(np.where(labels == 0)[0])
            print(f"False assignment percentage of member of capture cluster: {false_assignment_capture_percentage*100}")
            if len(np.where(labels == 1)[0]) > 0:
                false_assignment_escape_percentage = len(false_assignments_escape) / len(np.where(labels == 1)[0])
                print(f"False assignment percentage of member of  escape cluster: {false_assignment_escape_percentage*100}")
            else:
                false_assignment_escape_percentage = np.nan
                print("No members in escape cluster, setting false assignment percentage to 0")
            if len(np.where(labels == 2)[0]) > 0:
                false_assignment_crash_percentage = len(false_assignments_crash) / len(np.where(labels == 2)[0])
                print(f"False assignment percentage of member of crash cluster: {false_assignment_crash_percentage*100}")
            else:
                false_assignment_crash_percentage = np.nan
                print("No members in crash cluster, setting false assignment percentage to 0")

            false_capture_percent[ll, nn] = false_assignment_capture_percentage
            false_escape_percent[ll, nn] = false_assignment_escape_percentage
            false_crash_percent[ll, nn] = false_assignment_crash_percentage


            if saveLoopFigs:
                # Plot input data energy colored by assigned cluster (pred_label)
                fig, ax = plt.subplots()
                for ii in range(Nsamples):
                    if labels[ii] == 0:  # Capture
                        if pred_labels[ii] != labels[ii]:
                            ax.plot(samples[ii], color='C1', alpha=0.5)
                        else:
                            ax.plot(samples[ii], color='C0', alpha=0.5)
                    elif labels[ii] == 1:  # Escape
                        if pred_labels[ii] != labels[ii]:
                            ax.plot(samples[ii], color='C3', alpha=0.5)
                        else:
                            ax.plot(samples[ii], color='C2', alpha=0.5)
                    else:  # Crash
                        if pred_labels[ii] != labels[ii]:
                            ax.plot(samples[ii], color='C5', alpha=0.5)
                        else:
                            ax.plot(samples[ii], color='C4', alpha=0.5)
                ax.set_ylabel('Y Input')
                ax.set_xlabel('X Input')
                ax.plot([], color='C0', label='Correctly Predicted Capture')
                ax.plot([], color='C2', label='Correctly Predicted Escape')
                ax.plot([], color='C4', label='Correctly Predicted Impact')
                ax.plot([], color='C1', label='Incorrectly Predicted Capture')
                ax.plot([], color='C3', label='Incorrectly Predicted Escape')
                ax.plot([], color='C5', label='Incorrectly Predicted Impact')
                ax.axhline(0, color='black', linestyle='--')
                ax.legend(loc='lower left')
                plt.title(f"LD: {LD}, NC: {NC}")
                plt.tight_layout()
                plt.savefig(os.path.join(folder_path, f"predicted_clusters_{saveTag}_LD{LD}_NC{NC}.png"), dpi=300)
                # plt.show()

                fig, axs = plt.subplots(1,3, figsize=(12,4), sharey=True)
                for ii in range(Nsamples):
                    if labels[ii] == 0:  # Capture
                        if pred_labels[ii] == 1:  # Falsely assigned to escape
                            axs[0].plot(samples[ii], color='C6', alpha=0.5)
                        elif pred_labels[ii] == 2:  # Falsely assigned to impact
                            axs[0].plot(samples[ii], color='C7', alpha=0.5)
                        else:
                            axs[0].plot(samples[ii], color='C0', alpha=0.5)
                    elif labels[ii] == 1:  # Escape
                        if pred_labels[ii] == 0:  # Falsely assigned to capture
                            axs[1].plot(samples[ii], color='C8', alpha=0.5)
                        elif pred_labels[ii] == 2:  # Falsely assigned to impact
                            axs[1].plot(samples[ii], color='C9', alpha=0.5)
                        else:
                            axs[1].plot(samples[ii], color='C2', alpha=0.5)
                    else:  # Crash
                        if pred_labels[ii] == 0:  # Falsely assigned to capture
                            axs[2].plot(samples[ii], color='C10', alpha=0.5)
                        elif pred_labels[ii] == 1:  # Falsely assigned to escape
                            axs[2].plot(samples[ii], color='C11', alpha=0.5)
                        else:
                            axs[2].plot(samples[ii], color='C4', alpha=0.5)

                axs[0].plot([], color='C0', label='Correctly Predicted Capture')
                axs[0].plot([], color='C6', label='Assigned to Escape')
                axs[0].plot([], color='C7', label='Assigned to Impact')
                axs[1].plot([], color='C2', label='Correctly Predicted Escape')
                axs[1].plot([], color='C8', label='Assigned to Capture')
                axs[1].plot([], color='C9', label='Assigned to Impact')
                axs[2].plot([], color='C4', label='Correctly Predicted Impact')
                axs[2].plot([], color='C10', label='Assigned to Capture')
                axs[2].plot([], color='C11', label='Assigned to Escape')
                for ax in axs:
                    ax.set_ylabel('Y Input')
                    ax.set_xlabel('X Input')
                    ax.axhline(0, color='black', linestyle='--')
                    ax.legend(loc='lower left')
                plt.suptitle(f"LD: {LD}, NC: {NC}")
                plt.tight_layout()
                plt.savefig(os.path.join(folder_path, f"breakout_predicted_clusters_{saveTag}_LD{LD}_NC{NC}.png"), dpi=300)
                # plt.show()

    if len(LDs) == 1 or len(NCs) == 1:
        print(f'"encoderPathX": "{folder_path}",')
        print(f'"encoderSuffixX": "{suffix}", ')
        print(f'"captureIndsX": {[i for i, val in enumerate(assigned_cluster_inds) if val == 0]}, ')
        print(f'"escapeIndsX": {[i for i, val in enumerate(assigned_cluster_inds) if val == 1]}, ')
        print(f'"crashIndsX": {[i for i, val in enumerate(assigned_cluster_inds) if val == 2]}, ')

    if len(LDs) > 1 or len(NCs) > 1:
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

        # Print a booktabs latex table of the predicted crash probability for number of cluster and latent dimension
        print("Predicted Crash Probabilities")
        print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
        print("\\toprule")
        print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
        print("\\midrule")
        for ll, LD in enumerate(LDs):
            print(f"{LD} & " + " & ".join([f"{pred_crash_probs[ll, nn]:.4f}" for nn in range(len(NCs))]) + " \\\\")
        print("\\bottomrule")


        # Print capture percent misassignment
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\caption{Predicted Capture Misassignments}")
        print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
        print("\\toprule")
        print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
        print("\\midrule")
        for ll, LD in enumerate(LDs):
            print(f"{LD} & " + " & ".join([f"{false_capture_percent[ll, nn]*100:.4f}" for nn in range(len(NCs))]) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")

        # Print escape percent misassignment
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\caption{Predicted Escape Misassignments}")
        print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
        print("\\toprule")
        print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
        print("\\midrule")
        for ll, LD in enumerate(LDs):
            print(f"{LD} & " + " & ".join([f"{false_escape_percent[ll, nn]*100:.4f}" for nn in range(len(NCs))]) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")

        # Print crash percent misassignment
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\caption{Predicted Crash Misassignments}")
        print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
        print("\\toprule")
        print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
        print("\\midrule")
        for ll, LD in enumerate(LDs):
            print(f"{LD} & " + " & ".join([f"{false_crash_percent[ll, nn]*100:.4f}" for nn in range(len(NCs))]) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")

        # Print average percent misassignment
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\caption{Average Misassignments}")
        print("\\begin{tabular}{l" + "c" * len(NCs) + "}")
        print("\\toprule")
        print("Latent Dim & " + " & ".join([str(NC) for NC in NCs]) + " \\\\")
        print("\\midrule")
        for ll, LD in enumerate(LDs):
            print(f"{LD} & " + " & ".join(
                [f"{np.nanmean([false_crash_percent[ll, nn], false_escape_percent[ll, nn], false_capture_percent[ll, nn]]) * 100:.4f}" for nn in range(len(NCs))]) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")

        # Save false escape, false crash, and false capture percent to a npy file
        np.save(os.path.join(figPath, f"false_capture_percent_{saveTag}_{hds[0]}.npy"), false_capture_percent)
        np.save(os.path.join(figPath, f"false_escape_percent_{saveTag}_{hds[0]}.npy"), false_escape_percent)
        np.save(os.path.join(figPath, f"false_crash_percent_{saveTag}_{hds[0]}.npy"), false_crash_percent)

        plot_table_heatmap(
            false_capture_percent,
            LDs,
            NCs,
            f"{hds[0]}x{hds[1]}x{hds[2]}: False Capture Assignment (%)",
            save_path=os.path.join(figPath, f"heatmap_false_capture_{saveTag}_{hds[0]}.png"),
            cmap="Blues",
            percent=True
        )

        plot_table_heatmap(
            false_escape_percent,
            LDs,
            NCs,
            f"{hds[0]}x{hds[1]}x{hds[2]}: False Escape Assignment (%)",
            save_path=os.path.join(figPath, f"heatmap_false_escape_{saveTag}_{hds[0]}.png"),
            cmap="Greens",
            percent=True
        )

        plot_table_heatmap(
            false_crash_percent,
            LDs,
            NCs,
            f"{hds[0]}x{hds[1]}x{hds[2]}: False Impact Assignment (%)",
            save_path=os.path.join(figPath, f"heatmap_false_crash_{saveTag}_{hds[0]}.png"),
            cmap="Reds",
            percent=True
        )

        avg_false = np.nanmean(
            np.stack([false_capture_percent,
                      false_escape_percent,
                      false_crash_percent]),
            axis=0
        )

        plot_table_heatmap(
            avg_false,
            LDs,
            NCs,
            f"{hds[0]}x{hds[1]}x{hds[2]}: Average Misassignment (%)",
            save_path=os.path.join(figPath, f"heatmap_avg_false_{saveTag}_{hds[0]}.png"),
            cmap="flare",
            percent=True
        )

        # ----------------------------------------
        # Weighted average misassignment
        # ----------------------------------------
        # weighted_false = (
        #         capture_prob * false_capture_percent +
        #         escape_prob * false_escape_percent +
        #         impact_prob * false_crash_percent
        # )
        # Stack errors and weights
        errors = np.stack([false_capture_percent,false_escape_percent,false_crash_percent])

        weights = np.array([capture_prob,escape_prob,impact_prob])[:, None, None]  # shape (3,1,1) for broadcasting

        # Mask NaNs
        valid_mask = ~np.isnan(errors)

        # Zero-out invalid contributions
        weighted_sum = np.nansum(errors * weights * valid_mask, axis=0)

        # Renormalize weights where valid
        effective_weight = np.sum(weights * valid_mask, axis=0)

        weighted_false = weighted_sum / effective_weight

        plot_table_heatmap(
            weighted_false,
            LDs,
            NCs,
            f"{hds[0]}x{hds[1]}x{hds[2]}: Weighted Misassignment (%)",
            save_path=os.path.join(figPath, f"heatmap_weighted_false_{saveTag}_{hds[0]}.png"),
            cmap="crest",
            percent=True
        )

        # ----------------------------------------
        # Failure-only weighted misassignment
        # ----------------------------------------

        # failure_prob = escape_prob + impact_prob
        #
        # failure_weighted = (escape_prob * false_escape_percent + impact_prob * false_crash_percent) / failure_prob
        failure_errors = np.stack([false_escape_percent,false_crash_percent])

        failure_weights = np.array([escape_prob,impact_prob])[:, None, None]

        valid_mask = ~np.isnan(failure_errors)

        weighted_sum = np.nansum(failure_errors * failure_weights * valid_mask, axis=0)

        effective_weight = np.sum(failure_weights * valid_mask, axis=0)

        failure_weighted = weighted_sum / effective_weight

        plot_table_heatmap(
            failure_weighted,
            LDs,
            NCs,
            f"{hds[0]}x{hds[1]}x{hds[2]}: Failure Misassignment (%)",
            save_path=os.path.join(figPath, f"heatmap_failure_only_{saveTag}_{hds[0]}.png"),
            cmap="managua",
            percent=True
        )

        # print out the mean of the avergage, weighted average, and failure average over all LDs and NCs
        print(f"Mean Average Misassignment: {np.nanmean(avg_false)*100:.4f}%")
        print(f"Mean Weighted Misassignment: {np.nanmean(weighted_false)*100:.4f}%")
        print(f"Mean Failure-Weighted Misassignment: {np.nanmean(failure_weighted)*100:.4f}%")


if __name__ == "__main__":
    main()


