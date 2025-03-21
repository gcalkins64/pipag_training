import os
import re
import imageio

# Set your directory path
# LD 4 NC 2
# folder_path = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250320_170537"  # Change this to your actual folder
# LD 4 NC 3
# folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250320_192010'
# LD 4 NC 4
# folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250320_204111'
# LD 4 NC 5
# folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250320_223846'
# LD 4 NC 6
# folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250320_235238'
# LD 5 NC 2
folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250321_011131'
# LD 5 NC 3
# folder_path = '/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/gmvae_em_aerocapture_energy_20250321_011120'
# LD 5 NC 4

# LD 5 NC 5

# LD 5 NC 6

# LD 6 NC 2

# LD 6 NC 3

# LD 6 NC 4

# LD 6 NC 5

# LD 6 NC 6

# Regular expression to extract epoch number XXXXX from filename
pattern = re.compile(r"latent_samples_epoch_(\d+)_.*")

# Collect files with epoch numbers
image_files = []
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        epoch = int(match.group(1))  # Extract and convert XXXXX to an integer
        image_files.append((epoch, os.path.join(folder_path, filename)))

# Sort files by epoch number
image_files.sort()

# Extract just the sorted file paths
sorted_image_paths = [file[1] for file in image_files]

# Create a GIF
output_gif = os.path.join(folder_path, "latent_samples_animation.gif")
imageio.mimsave(output_gif, [imageio.imread(img) for img in sorted_image_paths], duration=0.1)  # Adjust duration as needed

print(f"GIF saved as {output_gif}")
