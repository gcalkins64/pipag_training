import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def main():
    sns.set_style('whitegrid')
    sns.set_palette("Set2")
    sns.set_context("notebook", rc={"lines.linewidth": 2.5, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 12,
                                    'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, "font.family": "serif"})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    # Load in sample dataset
    dataPath = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data'
    savePath = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs/SVD'
    fileNames = ['UOP_inc_lit_disps_5000_data_energy_scaled_downsampled_.json',
                 'UOP_near_crash_5000_data_energy_scaled_downsampled_.json',
                 'UOP_near_crash_steeper_5000_data_energy_scaled_downsampled_.json']
    tags = ['Near Escape', 'Near Crash', 'Aggressive Near Crash']
    
    all_samples = []
    # Load in all jsons
    for fileName in fileNames:
        with open(f'{dataPath}/{fileName}', 'r') as f:
            inputData = json.load(f)
        # Get all samples
        samples = np.array([inputData[f'sample{i}']['energy'] for i in range(len(inputData))])
        all_samples.append(samples)

    Ss = []
    # Plot the diagonals of S
    plt.figure(figsize=(5, 4))
    for samples in all_samples:
        # Perform SVD on samples
        samples = np.array(samples)
        samples = samples.T
        U, S, V = np.linalg.svd(samples, full_matrices=False)
        Ss.append(S)
        plt.plot(S, marker='o')
    plt.yscale('log')
    plt.title("Singular Values")
    plt.legend(tags)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(f'{savePath}/svd_singular_values.png')

    # Plot cumulative explained variance
    # Determine number of components to keep (e.g., 95% explained variance)
    threshold = 0.999
    plt.figure(figsize=(5, 4))
    for S in Ss:
        # Compute explained variance
        explained_variance = S ** 2 / np.sum(S ** 2)
        cumulative_explained_variance = np.cumsum(explained_variance)
        plt.plot(cumulative_explained_variance, marker='o')
        n_components = np.argmax(cumulative_explained_variance >= threshold) + 1
        print(f"Number of latent variables required to explain {threshold * 100}% variance: {n_components}")

    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Latent Variables")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=threshold, color='r', linestyle='--')
    tags.append(f'{threshold*100}% Threshold')
    plt.legend(tags)
    plt.tight_layout()
    plt.savefig(f'{savePath}/svd_cumulative_explained_variance.png')

    plt.show()


if __name__ == '__main__':
    main()