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



def create_batch(img_path, output_path, boundary_model, noise_lvls = [0.3,0.4], num_per_lvl = 10, img_shape = (150,150), img_name = ""):
    if not(os.path.exists(output_path) and os.path.isdir(output_path)):
        os.mkdir(output_path)

    clean_img = PIL.Image.open(gfile.GFile(img_path, 'rb')).resize(img_shape)

    #clean_img.save(f'{output_path}/clean_img.png')
    clean_img.save(f'{output_path}/clean_img_{img_name}.png')
    #print("saved clean image")

    clean_img = jnp.array(clean_img)/255.0

    input_clean_img = jnp.expand_dims(clean_img.transpose(2,0,1)[:3,:,:], axis=0)

    outputs_clean = boundary_model[0](boundary_model[1]['params'], input_clean_img)
    #print("computed clean image boundaries")

    clean_boundaries = outputs_clean[-1]['global_boundaries'].squeeze()

    img_out = PIL.Image.fromarray((np.array(clean_boundaries)*255).astype(np.uint8))

    img_out.save(f'{output_path}/clean_img_boundaries_{img_name}.png')
    #print("saved clean image boundaries")

    for sigma in noise_lvls:
        noise_fun = lambda x, s: jnp.clip(jnp.array(x + np.random.normal(0, s, x.shape)), 0, 1)

        if not(os.path.exists(f'{output_path}/noise_lvl_{sigma}/') and os.path.isdir(f'{output_path}/noise_lvl_{sigma}/')):
            os.mkdir(f'{output_path}/noise_lvl_{sigma}/')
        #print("Sigma lvl: ", sigma)
        for i in range(num_per_lvl):
            noisy_img = noise_fun(clean_img, sigma)

            input_noisy_img = jnp.expand_dims(noisy_img.transpose(2,0,1)[:3,:,:], axis=0)

            outputs_noisy = boundary_model[0](boundary_model[1]['params'], input_noisy_img)
            #print("noisy boundaries, ", i)

            noisy_boundaries = outputs_noisy[-1]['global_boundaries'].squeeze()

            img_out = PIL.Image.fromarray((np.array(noisy_boundaries)*255).astype(np.uint8))

            img_out.save(f'{output_path}/noise_lvl_{sigma}/img_boundaries_noisy_{img_name}_{i}.png')
            #print("saved noisy boundaries")


#create_batch('142.jpg', 'test/', (apply_jitted, trained_params), noise_lvls=[0.3,0.4], num_per_lvl=2, img_shape=(150,150))

print(jax.devices())


model = create_model_BA()
print("model created")

#files_path = [x for x in os.listdir("data/seg_train/seg_train/buildings/")]
#files_path = [x for x in os.listdir("Train/")]
#files_path = [x for x in os.listdir("lol_data/")]
files_path = [x for x in os.listdir("lol_unmerged_data/batch_2/low_2/")]

#print(files_path)

for i, img_name in enumerate(files_path):
    #print("starting batch gen")
    #create_batch(f'data/seg_train/seg_train/buildings/{img_name}', f'dataset/building_image_{i}', model, noise_lvls=[0.3,0.4,0.5], num_per_lvl=10, img_shape=(150,150))
    #create_batch(f'Train/{img_name}', f'dataset/obj_image_{i}', model, noise_lvls=[0.3,0.4,0.5, 0.6], num_per_lvl=10, img_shape=(150,150))
    #create_batch(f'lol_data/{img_name}', f'dataset/lol_image_{i}', model, noise_lvls=[0.3,0.4,0.5,0.6], num_per_lvl=10, img_shape=(150,150))
    create_batch(f'lol_unmerged_data/batch_2/low_2/{img_name}', f'dataset_lol/lol_image_batch_2_{i}', model, noise_lvls=[0.3,0.4,0.5], num_per_lvl=10, img_shape=(150,150), img_name=f'low_{img_name}')
    create_batch(f'lol_unmerged_data/batch_2/high_2/{img_name}', f'dataset_lol/lol_image_batch_2_{i}', model, noise_lvls=[0.3,0.4,0.5], num_per_lvl=10, img_shape=(150,150), img_name=f'high_{img_name}')
    print("Produced data for image: ", i)

