'''
Author: Cancan Huang
Date: 2021-01-27 02:29:30
LastEditTime: 2021-02-02 03:24:05
'''
# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale, resize
from PIL import Image
import math


def vis_hybrid_image(hybrid_image):
    """
    Visualize a hybrid image by progressively downsampling the image and
    concatenating all of the images together.
    """
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales + 1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))
        # downsample image
        cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel=True)
        # pad the top to append to the output
        pad = np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))
    return output


def load_image(path):
    return img_as_float32(io.imread(path))


def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))


# given two differently sized images, resize them so they have the same shape
def equalize_image_sizes(im_one, im_two):
    assert im_one.shape[2] == im_two.shape[2], 'the third dimension of these images do not match'
    # resizes by adding/subtracting half of the difference between the image's width and height
    x_resize = (im_one.shape[0] - im_two.shape[0]) / 2
    y_resize = (im_one.shape[1] - im_two.shape[1]) / 2
    im_one = resize(im_one, (int(im_one.shape[0] - x_resize), int(im_one.shape[1] - y_resize), im_one.shape[2]))
    im_two = resize(im_two, (int(im_two.shape[0] + x_resize), int(im_two.shape[1] + y_resize), im_two.shape[2]))
    return im_one, im_two
