import os
import shutil
import re
import json

# Define paths
basePath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/"
# destination_folder = "/Users/gracecalkins/Library/CloudStorage/OneDrive-UCB-O365/LaTeX/My_Lab_Notebook_25_01/figures"
# destination_folder = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs/gmvae_comparison_rev_gumixture_energy_biased_down_30"
# destination_folder = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs/gmvae_comparison_rev_gumixture_energy_biased_down_24"
destination_folder = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/figs/gmvae_comparison_rev_gumixture_energy_biased_down_36"

# LDs = [4,5,6]
LDs = [3,4,5,6,7,8,9]
NCs = [3,4,5,6,7,8,9]
# hds = [30,20,10]
# hds = [24,18,12]
hds = [36,24,12]

for LD in LDs:
    for NC in NCs:
        print(f"Processing LD: {LD}, NC: {NC}")

        # pattern = rf"^gmvae_em_aerocapture_energy_(20250429|20250430)_\d{{6}}_{LD}_{NC}$"
        # pattern = rf"^gmvae_near_escape_(20250527|20250528)_\d{{6}}_L{LD}_C{NC}$"
        # pattern = rf"^gmvae_near_crash_(20250528|20250529)_\d{{6}}_L{LD}_C{NC}$"
        # pattern = rf"^gmvae_near_escape_new_20250601_\d{{6}}_L{LD}_C{NC}$"
        # pattern = rf"^gmvae_uniform_new_20250601_\d{{6}}_L{LD}_C{NC}$"
        # pattern = rf"^gmvae_near_crash_new_20250601_\d{{6}}_L{LD}_C{NC}$"

        pattern = rf"^gmvae_rev_gu_mixture_energy_biased_down_\d{{8}}_\d{{6}}_L{LD}_C{NC}_{hds[0]}_{hds[1]}_{hds[2]}$"

        matching_folders = [
            f for f in os.listdir(basePath)
            if os.path.isdir(os.path.join(basePath, f)) and re.fullmatch(pattern, f)
        ]
        if matching_folders:
            print(f"LD {LD}, NC {NC} → {matching_folders}")
        else:
            print(f"LD {LD}, NC {NC} → no match")

        if not matching_folders:
            print(f"Warning: No folder found for LD{LD}_NC{NC}")
            continue

        folder_path = os.path.join(basePath, matching_folders[0])  # Assuming only one match

        # Define the file path
        # file_name = f"generated_samples_LD{LD}_NC{NC}.png"
        file_name = f"predicted_latent_clusters_rev_gumixture_energy_biased_down_LD{LD}_NC{NC}.png"
        # file_name = f"predicted_clusters_uni_LD{LD}_NC{NC}.png"
        file_path = os.path.join(folder_path, file_name)

        # Check if file exists before moving
        if os.path.exists(file_path):
            # NEW: append _hds[0] before extension
            new_file_name = (
                f"predicted_latent_clusters_rev_gumixture_energy_biased_down_"
                f"LD{LD}_NC{NC}_{hds[0]}.png"
            )

            shutil.copy(file_path, os.path.join(destination_folder, new_file_name))

            print(f"🍀 Copied as {new_file_name} to {destination_folder}")
        else:
            print(f"🚨 File {file_name} not found in {folder_path}")

print("File moving process completed.")
