from PIL import Image
import numpy as np

import os

from skimage import metrics

import re


def get_psnrs(out_folder = "output_overlay_nonadditive", post_threshold = 0.3, post_mult = 1):

    psnrs = [] # (id, psnr)

    psnrs_postprocessed = [] # (id, psnr)

    psnrs_input = [] # (id, psnr_avg, psnr_max)

    psnrs_input_postprocessed = [] # (id, psnr_avg, psnr_max)

    pat = re.compile(r'.*.png$')

    #iterate over overay 
    for idx in range(int(len(list(filter(pat.match, os.listdir(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/{out_folder}/')))) / 2)):

        #paths to output, clean image, and input folder (change these as needed)
        output_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/{out_folder}/test_{idx}_output_image.png'
        clean_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/{out_folder}/test_{idx}_clean_image.png'

        inputs_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/{out_folder}/input_imgs_{idx}/'

        #model output
        img_out = np.array(Image.open(output_path).resize((150,150))) / 255

        #model input
        img_clean = np.array(Image.open(clean_path).resize((150,150))) / 255


        psnrs_noisy = []

        psnrs_noisy_post = []

        for i in range(len(os.listdir(inputs_path))):
            img_in = np.array(Image.open(f'{inputs_path}test_{idx}_input_{i}.png').resize((150,150))) / 255

            #get input psnr
            psnrs_noisy.append(metrics.peak_signal_noise_ratio(img_in, img_clean))

            #post processed
            img_in = np.clip(post_mult * img_in * (img_in > post_threshold), 0, 1)

            psnrs_noisy_post.append(metrics.peak_signal_noise_ratio(img_in, img_clean))


        # get PSNR
        psnr_out = metrics.peak_signal_noise_ratio(img_out, img_clean)

        psnrs.append((idx, psnr_out))

        # postprocessing psnrs
        img_out = np.clip(post_mult * img_out * (img_out > post_threshold), 0, 1)

        psnr_out_post = metrics.peak_signal_noise_ratio(img_out, img_clean)

        psnrs_postprocessed.append((idx, psnr_out_post))

        # Get average and max input psnr
        psnr_noisy_avg = sum(psnrs_noisy) / len(psnrs_noisy)

        psnr_noisy_post_avg = sum(psnrs_noisy_post) / len(psnrs_noisy_post)

        psnr_noisy_max = max(psnrs_noisy)


        # postprocessed input psnrs
        psnr_noisy_post_max = max(psnrs_noisy_post)

        psnrs_input.append((idx, psnr_noisy_avg, psnr_noisy_max))

        psnrs_input_postprocessed.append((idx, psnr_noisy_post_avg, psnr_noisy_post_max))


    return (psnrs, psnrs_postprocessed, psnrs_input, psnrs_input_postprocessed)
    

#psnrs, psnrs_postprocessed, psnrs_input, psnrs_input_postprocessed = get_psnrs(out_folder="Report_Outputs/3_chan_noises/report_output_3chan_overlay_0.5_noise_model_1", post_threshold = 0.15, post_mult = 1)
psnrs, psnrs_postprocessed, psnrs_input, psnrs_input_postprocessed = get_psnrs(out_folder="Report_Outputs/report_output_model_13_on_additive_2", post_threshold = 0.15, post_mult = 1)
#psnrs, psnrs_postprocessed, psnrs_input, psnrs_input_postprocessed = get_psnrs(out_folder="report_output_model_14", post_threshold = 0.15, post_mult = 1)

#output results

print("Average PSNR: ", sum(x[1] for x in psnrs) / len(psnrs))

print("Average PSNR over inputs: ", sum(x[1] for x in psnrs_input) / len(psnrs_input))