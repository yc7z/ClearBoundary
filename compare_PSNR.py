from PIL import Image
import numpy as np

import os

from skimage import metrics

import re


def get_psnrs(out_num = 1, post_threshold = 0.3, post_mult = 1):

    psnrs = [] # (id, psnr)

    psnrs_postprocessed = [] # (id, psnr)

    psnrs_input = [] # (id, psnr_avg, psnr_max)

    psnrs_input_postprocessed = [] # (id, psnr_avg, psnr_max)

    pat = re.compile(r'.*.png$')

    #print(len(list(filter(pat.match, os.listdir(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/output_{out_num}/')))))
    for idx in range(int(len(list(filter(pat.match, os.listdir(f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/output_{out_num}/')))) / 2)):

        test_num = idx

        output_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/output_{out_num}/test_{test_num}_output_image.png'
        clean_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/output_{out_num}/test_{test_num}_clean_image.png'

        inputs_path = f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/output_{out_num}/input_imgs_{test_num}/'

        img_out = np.array(Image.open(output_path).resize((150,150))) / 255

        img_clean = np.array(Image.open(clean_path).resize((150,150))) / 255


        psnrs_noisy = []

        psnrs_noisy_post = []

        for i in range(len(os.listdir(inputs_path))):
            img_in = np.array(Image.open(f'{inputs_path}test_{test_num}_input_{i}.png').resize((150,150))) / 255

            psnrs_noisy.append(metrics.peak_signal_noise_ratio(img_in, img_clean))

            img_in = np.clip(post_mult * img_in * (img_in > post_threshold), 0, 1)

            psnrs_noisy_post.append(metrics.peak_signal_noise_ratio(img_in, img_clean))


        psnr_out = metrics.peak_signal_noise_ratio(img_out, img_clean)

        psnrs.append((idx, psnr_out))

        img_out = np.clip(post_mult * img_out * (img_out > post_threshold), 0, 1)

        #Image.fromarray((img * 255).astype(np.uint8)).save("output.png")

        psnr_out_post = metrics.peak_signal_noise_ratio(img_out, img_clean)

        psnrs_postprocessed.append((idx, psnr_out_post))

        #print("Model output PSNR: ", psnr_out)

        #print("Model output PSNR (with post processing): ", psnr_out_post)

        psnr_noisy_avg = sum(psnrs_noisy) / len(psnrs_noisy)

        psnr_noisy_post_avg = sum(psnrs_noisy_post) / len(psnrs_noisy_post)

        psnr_noisy_max = max(psnrs_noisy)

        psnr_noisy_post_max = max(psnrs_noisy_post)

        psnrs_input.append((idx, psnr_noisy_avg, psnr_noisy_max))

        psnrs_input_postprocessed.append((idx, psnr_noisy_post_avg, psnr_noisy_post_max))

        #print("Input Avg PSNR: ", psnr_noisy_avg)

        #print("Input Avg PSNR (with post processing): ", psnr_noisy_post_avg)

        #print("Input Max PSNR: ", psnr_noisy_max)

        #print("Input Max PSNR (with post processing): ", psnr_noisy_post_max)

    return (psnrs, psnrs_postprocessed, psnrs_input, psnrs_input_postprocessed)
    

psnrs, psnrs_postprocessed, psnrs_input, psnrs_input_postprocessed = get_psnrs(out_num=7, post_threshold = 0.3, post_mult = 1)

if False:
    for idx in range(11):
        print("---------------- IDX:", idx, "------------------")
        print("Output PSNR's")
        print(psnrs[idx])
        print("Output processed PSNR's")
        print(psnrs_postprocessed[idx])
        print("Input PSNR's")
        print(psnrs_input[idx])
        print("Input processed PSNR's")
        print(psnrs_input_postprocessed[idx])


print("Average PSNR: ", sum(x[1] for x in psnrs) / len(psnrs))
print("Average PSNR (postprocessed): ", sum(x[1] for x in psnrs_postprocessed) / len(psnrs_postprocessed))
print("Average PSNR over inputs: ", sum(x[1] for x in psnrs_input) / len(psnrs_input))
print("Average PSNR (postprocessed) over inputs: ", sum(x[1] for x in psnrs_input_postprocessed) / len(psnrs_input_postprocessed))