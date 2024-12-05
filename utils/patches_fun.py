
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms


#https://stackoverflow.com/questions/68150248/how-to-extract-overlapping-patches-from-a-3d-volume-and-recreate-the-input-shape

def extract_patches_2ds(x, kernel_size, padding=0, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)

    channels = x.shape[1]

    x = torch.nn.functional.pad(x, padding)
    # (B, C, H, W)
    x = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    # (B, C, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1])
    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1])
    # (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
    return x

def combine_patches_2d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    h_dim_out, w_dim_out = output_shape[2:]
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])

    # (B * h_dim_in * w_dim_in, C, kernel_size[0], kernel_size[1])
    x = x.view(-1, channels, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
    # (B, C, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
    x = x.permute(0,1,4,5,2,3)
    # (B, C, kernel_size[0], kernel_size[1], h_dim_in, w_dim_in)
    x = x.contiguous().view(-1, channels * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
    x = torch.nn.functional.fold(x, (h_dim_out, w_dim_out), kernel_size=(kernel_size[0], kernel_size[1]), padding=padding, stride=stride, dilation=dilation)
    # (B, C, H, W)
    return x
