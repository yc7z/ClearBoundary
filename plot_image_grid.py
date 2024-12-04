import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image

def plot_images_from_folders(max_noisy_images: int, root_dir, save_path="visualization.png"):
    """
    Plots images from subfolders in a grid with labeled columns.
    
    Each row contains images from one folder in the order:
    [original image, noisy boundaries, clean boundaries, model output].
    
    Args:
        max_noisy_images (int): Number of noisy images to plot.
        root_dir (str): The root directory containing subfolders.
        save_path (str): The path to save the resulting grid image.
    """
    # Collect all subfolders
    subfolders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    subfolders.sort()  # Optional: Sort for consistent order

    rows = []
    
    for folder in subfolders:
        # Define paths for the required images
        original_image = os.path.join(folder, "clean_img.png")
        clean_boundary = os.path.join(folder, "clean_boundaries.png")
        model_output = os.path.join(folder, "output_boundaries.png")
        noisy_images = sorted(glob(os.path.join(folder, "noisy_boundaries", "*.png")))

        # Load images
        images = []
        if os.path.exists(original_image):
            images.append(Image.open(original_image))
        images.extend([Image.open(img) for img in noisy_images[:max_noisy_images]])  # Limit noisy images
        if os.path.exists(clean_boundary):
            images.append(Image.open(clean_boundary))
        if os.path.exists(model_output):
            images.append(Image.open(model_output))

        rows.append(images)

    # Determine the grid size
    num_cols = 3 + max_noisy_images  # 1 for original, k for noisy, 1 for clean, 1 for output
    num_rows = len(rows)

    # Define column labels
    column_labels = (
        ["Original Image"]
        + [f"Noisy Boundaries {i+1}" for i in range(max_noisy_images)]
        + ["Clean Boundaries", "Model Output"]
    )

    # Create the figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    if num_rows == 1:
        axes = [axes]  # Make sure axes is iterable

    # Add column labels to the first row of axes
    for j, label in enumerate(column_labels):
        ax = axes[0][j] if num_rows > 1 else axes[j]
        ax.set_title(label, fontsize=12, pad=10)

    # Plot images
    for i, row in enumerate(rows):  # Rows correspond directly to subfolders
        for j in range(num_cols):
            ax = axes[i][j] if num_rows > 1 else axes[j]
            if j < len(row):
                ax.imshow(row[j], cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')  # Hide extra axes

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Grid visualization saved at: {save_path}")


# Example usage
root_directory = "plotting_test/example_images"
output_path = "grid_visualization_with_labels.png"
plot_images_from_folders(root_directory, save_path=output_path)
