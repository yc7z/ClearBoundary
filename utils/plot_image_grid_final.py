import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Directory containing the subfolders with images
root_dir = '/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/samples_for_plot_multiple_levels/'  # Replace with the path to your folder

# Create the labels for the columns
column_labels = ['clean_image', 'noisy_boundaries_1', 'noisy_boundaries_2', 'noisy_boundaries_3', 'clean_boundaries', 'overlap_patches', 'non_overlap_patches']

# Get a list of subfolders
subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# Set up the figure and axes
fig, axes = plt.subplots(len(subfolders), 7, figsize=(12, len(subfolders) * 2))  # Reduce figure width slightly to bring columns closer

# Loop through each subfolder and plot the images
for i, subfolder in enumerate(subfolders):
    # Get the image files in the subfolder
    subfolder_path = os.path.join(root_dir, subfolder)
    image_files = os.listdir(subfolder_path)
    
    # Initialize a list to hold the images
    images = [None] * 7
    
    # Loop through the image files and assign them based on the name pattern
    for image_file in image_files:
        image_path = os.path.join(subfolder_path, image_file)
        try:
            if 'clean_img' in image_file:
                images[0] = np.asarray(Image.open(image_path))
            elif 'noisy' in image_file:
                if images[1] is None:
                    images[1] = np.asarray(Image.open(image_path))
                elif images[2] is None:
                    images[2] = np.asarray(Image.open(image_path))
                elif images[3] is None:
                    images[3] = np.asarray(Image.open(image_path))
            elif 'clean_boundary' in image_file:
                images[4] = np.asarray(Image.open(image_path))
            elif 'overlap' in image_file and 'no' not in image_file:
                images[5] = np.asarray(Image.open(image_path))
            elif 'no_overlap' in image_file:
                images[6] = np.asarray(Image.open(image_path))
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
    
    # Plot the images in the appropriate columns
    for j, image in enumerate(images):
        if image is not None:  # Only plot if the image is valid
            axes[i, j].imshow(image, cmap='gray')
        else:
            # If the image is None, plot an empty frame with a placeholder text
            axes[i, j].axis('off')
            axes[i, j].text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=8)
        axes[i, j].axis('off')
        if i == 0:  # Add column labels only in the first row
            axes[i, j].set_title(column_labels[j], fontsize=10)

# Adjust layout with `hspace` and tighter `wspace`
plt.subplots_adjust(hspace=-0.6, wspace=0.02)  # Minimize column spacing
plt.tight_layout(pad=0.5)  # Ensure tight fit for the grid

# Save the figure
plt.savefig('final_grid_multiple_levels.png', bbox_inches='tight', pad_inches=0.1)
