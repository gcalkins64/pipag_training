import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
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

# dp values
dp_values = [1.5,1.6,1.7,1.8,1.9,2]
dp_strings = [str(dp).replace(".", "") for dp in dp_values]

# ----------------------------------------
# Storage dictionaries
# ----------------------------------------

results = {
    "capture": {pair: [] for pair in pairs},
    "escape": {pair: [] for pair in pairs},
    "crash":  {pair: [] for pair in pairs},
}

# ----------------------------------------
# Load arrays and extract values
# ----------------------------------------

for dp_str in dp_strings:

    folder_name = f"gmvae_comparison_rev_gumixture_energy_biased_down_eval_gaussian_dp{dp_str}_36"
    folder_path = os.path.join(base_path, folder_name)

    capture_file = os.path.join(
        folder_path,
        f"false_capture_percent_rev_gumixture_energy_biased_down_eval_gaussian_dp{dp_str}_36.npy"
    )

    escape_file = os.path.join(
        folder_path,
        f"false_escape_percent_rev_gumixture_energy_biased_down_eval_gaussian_dp{dp_str}_36.npy"
    )

    crash_file = os.path.join(
        folder_path,
        f"false_crash_percent_rev_gumixture_energy_biased_down_eval_gaussian_dp{dp_str}_36.npy"
    )

    if not os.path.exists(capture_file):
        print(f"Missing files for dp={dp_str}")
        continue

    capture_arr = np.load(capture_file)
    escape_arr  = np.load(escape_file)
    crash_arr   = np.load(crash_file)

    for (LD, NC) in pairs:
        ld_index = LDs.index(LD)
        nc_index = NCs.index(NC)

        results["capture"][(LD,NC)].append(capture_arr[ld_index, nc_index])
        results["escape"][(LD,NC)].append(escape_arr[ld_index, nc_index])
        results["crash"][(LD,NC)].append(crash_arr[ld_index, nc_index])

# ----------------------------------------
# Plotting
# ----------------------------------------

for metric in ["capture", "escape", "crash"]:
    plt.figure()

    for pair in pairs:
        plt.plot(dp_values, results[metric][pair], marker="o", label=f"LD={pair[0]}, NC={pair[1]}")

    plt.xlabel("dp")
    plt.ylabel("False Assignment Rate")
    plt.title(f"{metric.capitalize()} Misassignment vs dp")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"{metric}_misassignment_vs_dp.png"), dpi=300)

plt.show()