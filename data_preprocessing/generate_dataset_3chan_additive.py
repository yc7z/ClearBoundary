import PIL
 
import numpy as np
import os


def create_batch(img_path, output_path, in_noise = 0, noise_lvls = [0.3,0.4], num_per_lvl = 10, img_shape = (150,150)):
    if not(os.path.exists(output_path) and os.path.isdir(output_path)):
        os.mkdir(output_path)

    clean_img = PIL.Image.open(img_path).resize(img_shape)

    clean_img.save(f'{output_path}/clean_img.png')

    clean_img = np.array(clean_img)/255.0

    clean_img = np.clip(np.array(clean_img + np.random.normal(0, in_noise, clean_img.shape)), 0, 1)

    PIL.Image.fromarray((np.array(clean_img)*255).astype(np.uint8)).save(f'{output_path}/input_img.png')

    for sigma in noise_lvls:
        noise_fun = lambda x, s: np.clip(np.array(x + np.random.normal(0, s, x.shape)), 0, 1)

        if not(os.path.exists(f'{output_path}/noise_lvl_{sigma}/') and os.path.isdir(f'{output_path}/noise_lvl_{sigma}/')):
            os.mkdir(f'{output_path}/noise_lvl_{sigma}/')
        for i in range(num_per_lvl):
            noisy_img = noise_fun(clean_img, sigma)

            PIL.Image.fromarray((np.array(noisy_img)*255).astype(np.uint8)).save(f'{output_path}/noise_lvl_{sigma}/img_noisy_{i}.png')


files_path = [x for x in os.listdir("/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/seg_train/seg_train/buildings/")]


for i, img_name in enumerate(files_path):
    create_batch(f'/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/seg_train/seg_train/buildings/{img_name}', f'/scratch/ssd004/scratch/yuchongz/clear_boundary_artifacts/additional_noisy_data/building_image_{i}', in_noise = 0.3, noise_lvls=[0.05, 0.1, 0.15], num_per_lvl=30, img_shape=(150,150))
    print("Produced data for image: ", i, "/", len(files_path), "path: ", img_name)
