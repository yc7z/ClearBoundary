import PIL
import jax.numpy as jnp
from tensorflow.io import gfile
from scenic.projects.boundary_attention.configs import base_config
from scenic.projects.boundary_attention.helpers import train_utils
from scenic.projects.boundary_attention.helpers import viz_utils

import tensorflow_datasets as tfds
import scenic.projects.boundary_attention.kaleidoshapes.kaleidoshapes

import ml_collections
import jax

from scenic.projects.boundary_attention.dataset_lib import dataloader
from scenic.projects.boundary_attention.configs import kaleidoshapes_config
from scenic.projects.boundary_attention.dataset_lib.datasets import kaleidoshapes_dataset

import matplotlib.pyplot as plt
 
import numpy as np
#from skimage.filters import gaussian
import scipy.stats as stats
from os.path import join
import os


def create_model_BA(weights_dir = 'scenic/projects/boundary_attention/pretrained_weights/', 
                    imshape = (150,150,3)):
    config = base_config.get_config(model_name='boundary_attention',
                                  dataset_name='testing',
                                  input_size=imshape)
    return train_utils.make_apply(config, weights_dir) # tuple: (apply_jitted, trained_params)



def create_batch(img_path, output_path, in_noise = 0, noise_lvls = [0.3,0.4], num_per_lvl = 10, img_shape = (150,150)):
    if not(os.path.exists(output_path) and os.path.isdir(output_path)):
        os.mkdir(output_path)

    clean_img = PIL.Image.open(gfile.GFile(img_path, 'rb')).resize(img_shape)

    clean_img.save(f'{output_path}/clean_img.png')

    clean_img = np.array(clean_img)/255.0

    clean_img = np.clip(np.array(clean_img + np.random.normal(0, in_noise, clean_img.shape)), 0, 1)

    PIL.Image.fromarray((np.array(clean_img)*255).astype(np.uint8)).save(f'{output_path}/input_img.png')

    for sigma in noise_lvls:
        noise_fun = lambda x, s: np.clip(np.array(x + np.random.normal(0, s, x.shape)), 0, 1)

        if not(os.path.exists(f'{output_path}/noise_lvl_{sigma}/') and os.path.isdir(f'{output_path}/noise_lvl_{sigma}/')):
            os.mkdir(f'{output_path}/noise_lvl_{sigma}/')
        #print("Sigma lvl: ", sigma)
        for i in range(num_per_lvl):
            noisy_img = noise_fun(clean_img, sigma)

            PIL.Image.fromarray((np.array(noisy_img)*255).astype(np.uint8)).save(f'{output_path}/noise_lvl_{sigma}/img_noisy_{i}.png')




files_path = [x for x in os.listdir("data/seg_train/seg_train/forest/")]


for i, img_name in enumerate(files_path):
    create_batch(f'data/seg_train/seg_train/forest/{img_name}', f'/home/tsnow/CSC_2529_Project/ClearBoundary-main/ClearBoundary-main/additive_noisy_data/forest_image_{i}', in_noise = 0.3, noise_lvls=[0.05, 0.1, 0.15], num_per_lvl=30, img_shape=(150,150))
    print("Produced data for image: ", i, "/", len(files_path), "path: ", img_name)

