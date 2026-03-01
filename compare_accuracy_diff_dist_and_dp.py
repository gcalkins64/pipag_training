import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import glob
sys.path.append("/Users/gracecalkins/Local_Documents/local_code/pipag/pipag_base")
from plotting import seabornSettings  # type: ignore

# ----------------------------------------
# Configuration
# ----------------------------------------

seabornSettings()
sns.set_palette("Paired")

base_path = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs"

LDs = [3,4,5,6,7,8,9]
NCs = [3,4,5,6,7,8,9]

# Pairs to extract
pairs = [(5,5), (6,6)]

datasets = ["gmvae_comparison_rev_gumixture_energy_biased_down_eval_gumixture_dp15_36",
            "gmvae_comparison_rev_gumixture_energy_biased_down_eval_gaussian_dp15_36",
            "gmvae_comparison_rev_gumixture_energy_biased_down_eval_gaussian_dp16_36",
            "gmvae_comparison_rev_gumixture_energy_biased_down_eval_gaussian_dp17_36",
            "gmvae_comparison_rev_gumixture_energy_biased_down_eval_gaussian_dp18_36",
            "gmvae_comparison_rev_gumixture_energy_biased_down_eval_gaussian_dp19_36",
            "gmvae_comparison_rev_gumixture_energy_biased_down_eval_gaussian_dp2_36",]
tags = ["GU Mixture (dp=1.5)", "Gaussian (dp=1.5)", "Gaussian (dp=1.6)", "Gaussian (dp=1.7)", "Gaussian (dp=1.8)", "Gaussian (dp=1.9)", "Gaussian (dp=2.0)"]

# ----------------------------------------
# Storage dictionaries
# ----------------------------------------

results = {
    "capture": {pair: [] for pair in pairs},
    "escape": {pair: [] for pair in pairs},
    "crash":  {pair: [] for pair in pairs},
    "weighted_false": {pair: [] for pair in pairs},
    "failure_only_false": {pair: [] for pair in pairs},
}

# ----------------------------------------
# Load arrays and extract values
# ----------------------------------------
# Store global minimum across all LDs and NCs for each dp
global_min = {
    "capture": [],
    "escape": [],
    "crash": [],
    "weighted_false": [],
    "failure_only_false": []
}
for dataset in datasets:
    folder_path = os.path.join(base_path, dataset)

    # Use a wildcard to match any prefix
    capture_pattern = os.path.join(folder_path, f"false_capture_percent_*_36.npy")
    escape_pattern = os.path.join(folder_path, f"false_escape_percent_*_36.npy")
    crash_pattern = os.path.join(folder_path, f"false_crash_percent_*_36.npy")
    weighted_pattern = os.path.join(folder_path, f"weighted_false_*_36.npy")
    failure_pattern = os.path.join(folder_path, f"failure_weighted_*_36.npy")

    # Grab the first matching file for each
    capture_files = glob.glob(capture_pattern)
    escape_files = glob.glob(escape_pattern)
    crash_files = glob.glob(crash_pattern)
    weighted_files = glob.glob(weighted_pattern)
    failure_files = glob.glob(failure_pattern)

    if not capture_files:
        print("No capture files found!")
    else:
        capture_file = capture_files[0]
        escape_file = escape_files[0]
        crash_file = crash_files[0]
        weighted_file = weighted_files[0]
        failure_only_file = failure_files[0]

        # Load the arrays
        capture_array = np.load(capture_file)
        escape_array = np.load(escape_file)
        crash_array = np.load(crash_file)
        weighted_array = np.load(weighted_file)
        failure_array = np.load(failure_only_file)

        print("Files loaded successfully:")
        print(capture_file, escape_file, crash_file)

    capture_arr = np.load(capture_file)
    escape_arr  = np.load(escape_file)
    crash_arr   = np.load(crash_file)
    weighted_arr = np.load(weighted_file)
    failure_only_arr = np.load(failure_only_file)

    for (LD, NC) in pairs:
        ld_index = LDs.index(LD)
        nc_index = NCs.index(NC)

        results["capture"][(LD,NC)].append(capture_arr[ld_index, nc_index])
        results["escape"][(LD,NC)].append(escape_arr[ld_index, nc_index])
        results["crash"][(LD,NC)].append(crash_arr[ld_index, nc_index])
        results["weighted_false"][(LD,NC)].append(weighted_arr[ld_index, nc_index])
        results["failure_only_false"][(LD,NC)].append(failure_only_arr[ld_index, nc_index])

    # Compute global minimum across all LDs and NCs
    for metric_name, arr in [
        ("capture", capture_arr),
        ("escape", escape_arr),
        ("crash", crash_arr),
        ("weighted_false", weighted_arr),
        ("failure_only_false", failure_only_arr),
    ]:
        if np.all(np.isnan(arr)):
            print("Array contains only NaNs, skipping min computation")
            min_index = None
            val = np.nan
            LD = 0
            NC = 0
        else:
            min_index = np.nanargmin(arr)
            print("Minimum at index:", min_index)

            ld_idx, nc_idx = np.unravel_index(min_index, arr.shape)
            val = arr[ld_idx, nc_idx] if min_index is not None else np.nan
            LD = LDs[ld_idx] if min_index is not None else None
            NC = NCs[nc_idx] if min_index is not None else None

        global_min[metric_name].append({
            "value": val,
            "LD": LD,
            "NC": NC
        })


# ----------------------------------------
# Plotting
# ----------------------------------------

for metric in ["capture", "escape", "crash", "weighted_false", "failure_only_false"]:
    plt.figure()

    for pair in pairs:
        # Convert to percent
        percent_values = 100 * np.array(results[metric][pair])

        # Plot curve
        plt.plot( percent_values,
                 marker="o",
                 label=f"LD={pair[0]}, NC={pair[1]}")

        # Make x ticks labeled with tags
        plt.xticks(range(len(percent_values)), tags)
        ax = plt.gca()
        ax.tick_params(axis='x', labelrotation=45)

        # Baseline (dp = 1.5 is first entry)
        baseline = percent_values[0]

        # Horizontal baseline line
        plt.axhline(
            y=baseline,
            linestyle="--",
            alpha=0.7,
            color="C0" if pair == (5,5) else "C1",
            label=f"Baseline GU Mixture for LD={pair[0]}, NC={pair[1]}"
        )

    # 🔴 Scatter global minimum across all LDs and NCs
    # Extract min values in percent
    min_percent = [100 * entry["value"] for entry in global_min[metric]]

    plt.scatter(range(len(min_percent)),
                min_percent,
                marker="*",
                s=180,
                zorder=5,
                label="Global Min (all LD, NC)",
                color="C5")

    # Add text annotations above stars
    for i, dp in enumerate(range(len(min_percent))):
        entry = global_min[metric][i]
        label_text = f"({entry['LD']},{entry['NC']})"

        plt.text(dp,
                 min_percent[i] + 0.2,  # vertical offset
                 label_text,
                 ha='center',
                 va='bottom',
                 fontsize=9,
                 color="C5")

    plt.xlabel("dp")
    plt.ylabel("False Assignment Rate")
    plt.title(f"{metric.capitalize()} Misassignment vs Dist/Dp")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"{metric}_misassignment_vs_distdp.png"), dpi=300)

plt.show()