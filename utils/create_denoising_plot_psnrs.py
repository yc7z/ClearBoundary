import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image

from skimage import metrics
import numpy as np

max_noisy_images = 3
num_rows = 5
num_cols = 7

images = []

psnrs_30 = []
psnrs_40 = []
psnrs_50 = []

avg_in_psnrs_30 = []
avg_in_psnrs_40 = []
avg_in_psnrs_50 = []

#load in images
for i in [66, 77,197,209,101]:
    img_num_1 = i

    clean_img_1 = Image.open(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.3_noise_model_1/test_{img_num_1}_clean_image.png')
    output_img_noise30_1 = Image.open(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.3_noise_model_1/test_{img_num_1}_output_image.png')
    input_img_noise30_1 = Image.open(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.3_noise_model_1/input_imgs_{img_num_1}/test_{img_num_1}_input_0.png')
    output_img_noise40_1 = Image.open(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.4_noise_model_1/test_{img_num_1}_output_image.png')
    input_img_noise40_1 = Image.open(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.4_noise_model_1/input_imgs_{img_num_1}/test_{img_num_1}_input_0.png')
    output_img_noise50_1 = Image.open(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.5_noise_model_1/test_{img_num_1}_output_image.png')
    input_img_noise50_1 = Image.open(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.5_noise_model_1/input_imgs_{img_num_1}/test_{img_num_1}_input_0.png')
    
    clean_psnr_img = np.array(clean_img_1) / 255

    psnrs_30.append(metrics.peak_signal_noise_ratio(np.array(output_img_noise30_1)/255, clean_psnr_img))
    psnrs_40.append(metrics.peak_signal_noise_ratio(np.array(output_img_noise40_1)/255, clean_psnr_img))
    psnrs_50.append(metrics.peak_signal_noise_ratio(np.array(output_img_noise50_1)/255, clean_psnr_img))

    temp = []

    # compute psnr for lvl 0.3
    inputs_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.3_noise_model_1/input_imgs_{img_num_1}/'
    for i in range(len(os.listdir(inputs_path))):
        img_in = np.array(Image.open(f'{inputs_path}test_{img_num_1}_input_{i}.png').resize((150,150))) / 255

        temp.append(metrics.peak_signal_noise_ratio(img_in, clean_psnr_img))

    avg_in_psnrs_30.append((sum(temp) / len(temp), max))

    temp = []

    # compute psnr for lvl 0.4
    inputs_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.4_noise_model_1/input_imgs_{img_num_1}/'
    for i in range(len(os.listdir(inputs_path))):
        img_in = np.array(Image.open(f'{inputs_path}test_{img_num_1}_input_{i}.png').resize((150,150))) / 255

        temp.append(metrics.peak_signal_noise_ratio(img_in, clean_psnr_img))

    avg_in_psnrs_40.append((sum(temp) / len(temp), max))

    temp = []

    # compute psnr for lvl 0.5
    inputs_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.5_noise_model_1/input_imgs_{img_num_1}/'
    for i in range(len(os.listdir(inputs_path))):
        img_in = np.array(Image.open(f'{inputs_path}test_{img_num_1}_input_{i}.png').resize((150,150))) / 255

        temp.append(metrics.peak_signal_noise_ratio(img_in, clean_psnr_img))

    avg_in_psnrs_50.append((sum(temp) / len(temp), max))

    images.append([clean_img_1, input_img_noise30_1, output_img_noise30_1, input_img_noise40_1, output_img_noise40_1, input_img_noise50_1, output_img_noise50_1])

# Determine the grid size

#column names
cols = ['Original Image', 'Noisy Image 0.3', 'Model Output 0.3', 'Noisy Image 0.4', 'Model Output 0.4', 'Noisy Image 0.5', 'Model Output 0.5']

# Create the figure
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))


#make plot
for r in range(num_rows):
    for c in range(num_cols):
        axes[r,c].imshow(images[r][c])
        axes[r,c].axis('off')
        if r == 0:
            axes[r,c].set_title(cols[c], fontsize=12, pad=10)


plt.tight_layout()

plt.savefig('/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/denoise_grid.png')


#output PSNR's for each noise level
for i in range(5):
    print("------------------------------------------------: ", i)
    #output PSNR
    print(psnrs_30[i])
    print(psnrs_40[i])
    print(psnrs_50[i])

    #input PSNR
    print(avg_in_psnrs_30[i])
    print(avg_in_psnrs_40[i])
    print(avg_in_psnrs_50[i])
