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
 
import numpy as np
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

    clean_img.save(f'{output_path}/clean_img_{img_name}.png')
    clean_img = jnp.array(clean_img)/255.0

    input_clean_img = jnp.expand_dims(clean_img.transpose(2,0,1)[:3,:,:], axis=0)

    outputs_clean = boundary_model[0](boundary_model[1]['params'], input_clean_img)
    clean_boundaries = outputs_clean[-1]['global_boundaries'].squeeze()

    img_out = PIL.Image.fromarray((np.array(clean_boundaries)*255).astype(np.uint8))

    img_out.save(f'{output_path}/clean_img_boundaries_{img_name}.png')
    print("saved clean image boundaries")

    for sigma in noise_lvls:
        noise_fun = lambda x, s: jnp.clip(jnp.array(x + np.random.normal(0, s, x.shape)), 0, 1)

        if not(os.path.exists(f'{output_path}/noise_lvl_{sigma}/') and os.path.isdir(f'{output_path}/noise_lvl_{sigma}/')):
            os.mkdir(f'{output_path}/noise_lvl_{sigma}/')
        for i in range(num_per_lvl):
            noisy_img = noise_fun(clean_img, sigma)

            input_noisy_img = jnp.expand_dims(noisy_img.transpose(2,0,1)[:3,:,:], axis=0)

            outputs_noisy = boundary_model[0](boundary_model[1]['params'], input_noisy_img)

            noisy_boundaries = outputs_noisy[-1]['global_boundaries'].squeeze()

            img_out = PIL.Image.fromarray((np.array(noisy_boundaries)*255).astype(np.uint8))

            img_out.save(f'{output_path}/noise_lvl_{sigma}/img_boundaries_noisy_{img_name}_{i}.png')


print(jax.devices())

model = create_model_BA()
print("model created")

files_path = [x for x in os.listdir("data/seg_train/seg_train/buildings/")]

for i, img_name in enumerate(files_path):
    print("starting batch gen")
    create_batch(f'data/seg_train/seg_train/buildings/{img_name}', f'dataset/building_image_{i}', model, noise_lvls=[0.3,0.4,0.5], num_per_lvl=10, img_shape=(150,150))
    print("Produced data for image: ", i)
