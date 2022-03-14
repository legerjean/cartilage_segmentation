from __future__ import print_function
import numpy as np
from skimage.transform import rescale
from skimage.morphology import disk
from skimage.measure import label, regionprops
from scipy import ndimage
import os
from PIL import Image
import imageio
from skimage.color import gray2rgb, rgb2gray, rgb2hsv, hsv2rgb
from skimage.transform import downscale_local_mean
from scipy.ndimage import measurements
import pickle
from core.utils_core import *
from imageio import imwrite


def force_min_height(image, mask, ht_min):
    final_sh = np.array(image.shape)
    if final_sh[2] < ht_min:
        npad = ((0, 0), (0, 0), (0, int(ht_min - final_sh[2])))
        image = np.pad(image, pad_width=npad, mode='edge')
        mask = np.pad(mask, pad_width=npad, mode='edge')
    return image, mask


def extract_3d_tile(image_volume, mask_volume, wd, ht_min=None):
    # Image dimensions
    sh = image_volume.shape
    wd2 = int(wd / 2.0)

    # Center of mass of the mask
    image_thr = np.zeros(image_volume.shape)
    image_thr[image_volume > 0.5 * 255] = 1
    center = measurements.center_of_mass(image_thr)
    center = np.array(center)
    center = center.astype(int)
    center[0] = min(max(wd2, center[0]), sh[0] - wd2)
    center[1] = min(max(wd2, center[1]), sh[1] - wd2)

    image_tile = image_volume[center[0] - wd2:center[0] - wd2 + wd, center[1] - wd2:center[1] - wd2 + wd, :]
    mask_tile = mask_volume[center[0] - wd2:center[0] - wd2 + wd, center[1] - wd2:center[1] - wd2 + wd, :]

    # Check for height
    if ht_min is not None:
        image_tile, mask_tile = force_min_height(image_tile, mask_tile, ht_min)

    return image_tile, mask_tile, center


def generate_cropped_datasets(path_src, path_dest, **kwargs):
    # Get kwargs and save empty dictionary
    folders_names = kwargs.get('folders_names', ['AM1', 'AM2', 'AM3', 'AM4', 'AM5', 'DIO2', 'DIO3', 'DIO4', 'DIO5',
                                                 'YOUNG1', 'YOUNG3', 'YOUNG4'])
    nb_datasets = kwargs.get('nb_datasets', 12)
    pickle_out = open(path_dest + 'centers.pickle', "wb")
    pickle.dump({}, pickle_out)
    pickle_out.close()

    for i in range(nb_datasets):
        # Sort raw jpeg and png images
        image_filesList = os.listdir(path_src + 'images/' + folders_names[i] + '/')
        image_filesList.sort()
        mask_filesList = os.listdir(path_src + 'cartilage_mask/' + folders_names[i] + '/BIN/')
        mask_filesList.sort()

        # Instantiate volume to store the 3d data
        nb_files = int(len(image_filesList))
        im = path_src + 'images/' + folders_names[i] + '/' + image_filesList[0]
        im = imageio.imread(im)
        im_shape = im.shape
        image_volume = np.zeros((im_shape[0], im_shape[1], nb_files))
        mask_volume = np.zeros((im_shape[0], im_shape[1], nb_files))

        # Fill in the 3d data
        for j in range(nb_files):
            image_file = path_src + 'images/' + folders_names[i] + '/' + image_filesList[j]
            image_rgb = imageio.imread(image_file)
            image_gray = rgb2gray(image_rgb)  # in [0,1] range
            image_volume[:, :, j] = (image_gray * 255).astype(int)
            mask_file = path_src + 'cartilage_mask/' + folders_names[i] + '/BIN/' + mask_filesList[j]
            mask_rgb = imageio.imread(mask_file)
            mask_gray = rgb2gray(mask_rgb)
            mask_volume[mask_gray > 0, j] = 1
            print(j)
        print(i)

        # Crop dataset and update centers
        image_tile, mask_tile, center = extract_3d_tile(image_volume, mask_volume, 1024)
        pickle_in = open(path_dest + 'centers.pickle', "rb")
        centers_dict = pickle.load(pickle_in)
        pickle_in.close()
        centers_dict[i] = center
        pickle_out = open(path_dest + 'centers.pickle', "wb")
        pickle.dump(centers_dict, pickle_out)
        pickle_out.close()

        # Save images
        image_tile = image_tile.astype(np.uint8)
        mask_tile = mask_tile.astype(np.uint8)
        np.save(path_dest + str(i) + '_image.npy', image_tile)
        np.save(path_dest + str(i) + '_labels.npy', mask_tile)


def generate_partition(path_src, path_dest, **kwargs):
    # Get kwargs
    nb_datasets = kwargs.get('nb_datasets', 12)
    image_partition_0_200 = np.zeros((1200, 1024, 1024))
    mask_partition_0_200 = np.zeros((1200, 1024, 1024))

    for i in range(nb_datasets):
        image = np.load(path_src + str(i) + '_image.npy')
        mask = np.load(path_src + str(i) + '_labels.npy')

        # Build partition
        for ind in range(100):
            image_partition_0_200[i * 100+ind, :, :] = image[:, :, int(ind*2)]
            mask_partition_0_200[i * 100+ind, :, :] = mask[:, :, int(ind*2)]

        # Save images
        image_partition_0_200_uint8 = image_partition_0_200.astype(np.uint8)
        mask_partition_0_200_uint8 = mask_partition_0_200.astype(np.uint8)
        np.save(path_dest + 'image_partition_0_200.npy', image_partition_0_200_uint8)
        np.save(path_dest + 'mask_partition_0_200.npy', mask_partition_0_200_uint8)



