import os
import shutil
import re
import json

# Define paths
basePath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/"
destination_folder = "/Users/gracecalkins/Library/CloudStorage/OneDrive-UCB-O365/LaTeX/My_Lab_Notebook_25_01/figures"

LDs = [4, 5, 6]
NCs = [2, 3, 4, 5, 6]

for LD in LDs:
    for NC in NCs:
        print(f"Processing LD: {LD}, NC: {NC}")

        pattern = rf"^gmvae_em_aerocapture_energy_(20250429|20250430)_\d{{6}}_{LD}_{NC}$"
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
        # file_name = f"predicted_latent_clusters_LD{LD}_NC{NC}.png"
        file_name = f"predicted_clusters_LD{LD}_NC{NC}.png"
        file_path = os.path.join(folder_path, file_name)

        # Check if file exists before moving
        if os.path.exists(file_path):
            shutil.copy(file_path, os.path.join(destination_folder, file_name))
            print(f"Moved {file_name} to {destination_folder}")
        else:
            print(f"File {file_name} not found in {folder_path}")

print("File moving process completed.")
