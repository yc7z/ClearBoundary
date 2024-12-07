# ClearBoundary

First, install the dependencies in your virtual environment

```
pip install torch torchvision numpy wandb
```

The next step is to use a boundary detection algorithm on some image dataset to generate the boundary results. In our experiments, we used the Boundary Attention model:
https://github.com/google-research/scenic/tree/main/scenic/projects/boundary_attention

The scripts for processing any image dataset using Boundary Attention can be found under `data_preprocessing`. The scripts first add noises to the input images then apply Boundary Attention.

After you have the boundary results saved as images, if you want to use the dataloader creation scripts under `dataloader_creation`, then you should make sure the boundary results are saved in the following directory structure:

```
- dataset
  - image1
    - clean_image.png
    - noise_level_1
      - noisy_boundary1.png
      - noisy_boundary2.png
      ....
    - noise_level_2
  - image2
  ...
```

Once you have this, then you can simply run `python -m main` to start training. To train on boundary results using overlapping patches, modify `main.py` to use `trainers.train_overlapping`.

We provide some checkpoints of our model at https://drive.google.com/drive/folders/1xcZgzVohyuVok8gsHjAS6lO_c9nWSNgc?usp=sharing
