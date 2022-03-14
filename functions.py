import os
import numpy as np
import keras
import random
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, skeletonize, erosion
import tensorflow as tf
from imageio import imwrite
from compute_metrics import *
import imageio
import cv2
import seaborn as sns
from sklearn.decomposition import PCA
from core.utils_core import *
from core.train import *
from core.predict import *
import csv
import pickle
from scipy.ndimage.filters import convolve
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

# from fusion import *


#######################################
# Bio stats computation               #
#######################################

def compute_threshold_3d(image_3d):
    sh = image_3d.shape
    thr_array = np.zeros(sh[-1])
    for i in range(sh[-1]):
        thr_array[i] = compute_threshold_2d(image_3d[:, :, i])

    return np.mean(thr_array)


def compute_threshold_2d(image_2d):
    # Fix threshold using the histogram-based method in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3870034/
    # blur = cv2.GaussianBlur(image_2d, (5, 5), 0)
    ret3, th3 = cv2.threshold(image_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return ret3


def compute_porosity(image_2d, mask_2d, thr):
    # Porosity
    image_thr_i = np.zeros(image_2d.shape)
    image_thr_i[image_2d < thr] = 1
    surface_pores_i = np.multiply(image_thr_i, mask_2d)
    porosity = np.sum(surface_pores_i) / np.sum(mask_2d)

    # Number of pores
    regions = regionprops(label(surface_pores_i))
    nb_pores = len(regions)

    return porosity, nb_pores


def compute_average_width(mask_2d):
    skeleton_i = skeletonize(mask_2d)
    se = disk(1)
    mask_eroded_i = erosion(mask_2d, se)
    mask_contour_i = mask_2d - mask_eroded_i
    coordinates_contour_i = np.asarray(np.nonzero(mask_contour_i))
    coordinates_skeleton_i = np.asarray(np.nonzero(skeleton_i))
    sh_skeleton = coordinates_skeleton_i.shape
    width_array = np.zeros(sh_skeleton[-1])
    for j in range(sh_skeleton[-1]):
        width_array[j] = 2 * np.min(np.sqrt(np.power(coordinates_skeleton_i[0][j] - coordinates_contour_i[0], 2)
                                            + np.power(coordinates_skeleton_i[1][j] - coordinates_contour_i[1], 2)))
    return np.mean(width_array)


def compute_dice_score(mask_2d_prediction, mask_2d_labels):
    tn, fp, fn, tp = confusion_matrix(mask_2d_prediction.flatten(), mask_2d_labels.flatten()).ravel()
    return 2 * tp / (tp + fp + tp + fn)


def compute_bio_stats(image_3d, mask_3d, folder_name, **kwargs):
    mask_3d_labels = kwargs.get('mask_3d_labels', None)
    thr = compute_threshold_3d(image_3d)
    sh = image_3d.shape

    surface = np.zeros(sh[-1])
    porosity = np.zeros(sh[-1])
    nb_pores = np.zeros(sh[-1])
    average_width = np.zeros(sh[-1])
    dsc = np.zeros(sh[-1])

    for i in range(sh[-1]):
        surface[i] = np.sum(mask_3d[:, :, i])
        porosity_i, nb_pores_i = compute_porosity(image_3d[:, :, i], mask_3d[:, :, i], thr)
        porosity[i] = porosity_i
        nb_pores[i] = nb_pores_i
        average_width[i] = compute_average_width(mask_3d[:, :, i])
        if mask_3d_labels is not None:
            dsc[i] = compute_dice_score(mask_3d[:, :, i], mask_3d_labels[:, :, i])
        print(i)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    np.save(folder_name + 'surface.npy', surface)
    np.save(folder_name + 'porosity.npy', porosity)
    np.save(folder_name + 'nb_pores.npy', nb_pores)
    np.save(folder_name + 'average_width.npy', average_width)
    if mask_3d_labels is not None:
        np.save(folder_name + 'dsc.npy', dsc)


def majority_voting_TTA_bio_stats():
    option_list = [15, 16, 17, 18, 19]
    for option in option_list:
        if option == 15:
            my_src_dir_list = [
                'model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_2dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_3dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_4dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_5dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_6dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_7dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_8dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_9dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_10dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_11dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_12dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_13dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_14dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_15dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_16dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_17dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_18dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_19dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_20dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_21dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_22dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_23dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_24dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_25dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            ]
            nTrain = 10
            letter = 'dataset0'
            shortname = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        elif option == 16:
            my_src_dir_list = [
                'model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_2d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_3d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_4d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_10d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            ]
            nTrain = 10
            letter = 'D'
            shortname = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        elif option == 17:
            my_src_dir_list = [
                'model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            ]
            nTrain = 40
            letter = 'CDEF'
            shortname = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        elif option == 18:
            my_src_dir_list = [
                'model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            ]
            nTrain = 800
            letter = 'CDEF'
            shortname = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        elif option == 19:
            my_src_dir_list = [
                'model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_26d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_27d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_28d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_29d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_30d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            ]
            nTrain = 200
            letter = 'D'
            shortname = [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

        for j, filename in enumerate(my_src_dir_list):
            for i, value in enumerate([6, 7, 9, 10]):
                image = np.load('./samples/raw/' + str(value) + '_image.npy')
                image = image[:, :, ::4]
                #mask_fusion_loaded = np.transpose(np.load('./results/' + filename + '/firstval600/predictions/' + str(value) +
                #                                          '_prediction_fusion_TTA_new_20predictions.npy'),
                #                                  (1, 2, 0))
                mask_fusion_loaded = np.transpose(np.load('/DATA/jeaneliott/cartilage_results/' + filename + '/predictions/' + str(value) +
                                                      '_prediction_fusion_TTA_new_20predictions.npy'),
                                                 (1, 2, 0))
                mask_fusion = np.zeros(mask_fusion_loaded.shape)
                mask_fusion[mask_fusion_loaded > 10] = 1
                folder_name = './metrics/' + letter + '/' + shortname[j] + '/' + str(nTrain) + '_TTA_hardfusion/' + str(value) + '/'
                mask_labels = np.load('./samples/raw/' + str(value) + '_labels.npy')
                if i == 7:
                    mask_labels[:, :, 79] = mask_labels[:, :, 78]
                    mask_labels[:, :, 80] = mask_labels[:, :, 81]
                mask_labels = mask_labels[:, :, ::4]
                compute_bio_stats(image, mask_fusion, folder_name, mask_3d_labels=mask_labels)


def run_compute_bio_stats_GT():
    for i, value in enumerate([6, 7, 9, 10]):
    #for i, value in enumerate([7]):
        image = np.load('/export/home/jleger/Documents/segmentation/cartilage/samples/raw/' + str(value) + '_image.npy')
        mask = np.load('/export/home/jleger/Documents/segmentation/cartilage/samples/raw/' + str(value) + '_labels.npy')
        # compute_bio_stats(image[::4, ::4, :], mask[::4, ::4, :], value)
        folder_name = './metrics/GT/' + str(value) + '/'
        if value == 7:
            mask[:, :, 79] = mask[:, :, 78]
            mask[:, :, 80] = mask[:, :, 81]
        compute_bio_stats(image[:, :, :], mask[:, :, :], folder_name)


########################
# Average biostats     #
########################

def average_TTA_bio_stats():

    nb_images = [10, 10, 200, 40, 800]
    folders = ['dataset0', 'D', 'D', 'CDEF', 'CDEF']
    init_ranges = [range(1, 26),
                   range(1, 26),
                   np.concatenate((np.arange(5, 10), np.arange(11, 31))),
                   range(1, 26),
                   range(1, 26)]

    for config in range(5):
        nb_im = nb_images[config]
        folder = folders[config]
        init_range = init_ranges[config]
        for i in init_range:
            print(i)
            path = './metrics/' + str(folder) + '/' + str(i) + '/'
            if not os.path.exists(path + str(nb_im) + '_TTA_mean'):
                os.makedirs(path + str(nb_im) + '_TTA_mean')
            for k in [6, 7, 9, 10]:
                metrics_0 = np.load(path + str(nb_im) + '_TTA' + str(0) + '/' + str(k) + '/surface.npy')
                length = len(metrics_0)
                surface_800_array = np.zeros((20, length))
                nb_pores_800_array = np.zeros((20, length))
                porosity_800_array = np.zeros((20, length))
                average_width_800_array = np.zeros((20, length))

                for j in range(0, 20):
                    surface_800_array[j] = np.load(path + str(nb_im) + '_TTA' + str(j) + '/' + str(k) + '/surface.npy')
                    nb_pores_800_array[j] = np.load(path + str(nb_im) + '_TTA' + str(j) + '/' + str(k) + '/nb_pores.npy')
                    porosity_800_array[j] = np.load(path + str(nb_im) + '_TTA' + str(j) + '/' + str(k) + '/porosity.npy')
                    average_width_800_array[j] = np.load(path + str(nb_im) + '_TTA' + str(j) + '/' + str(k) + '/average_width.npy')

                surface_800_mean = np.mean(surface_800_array, axis=0)
                nb_pores_800_mean = np.mean(nb_pores_800_array, axis=0)
                porosity_800_mean = np.mean(porosity_800_array, axis=0)
                average_width_800_mean = np.mean(average_width_800_array, axis=0)

                if not os.path.exists(path + str(nb_im) + '_TTA_mean' + '/' + str(k)):
                    os.makedirs(path + str(nb_im) + '_TTA_mean' + '/' + str(k))

                np.save(path + str(nb_im) + '_TTA_mean' + '/' + str(k) + '/' + 'surface.npy', surface_800_mean)
                np.save(path + str(nb_im) + '_TTA_mean' + '/' + str(k) + '/' + 'nb_pores.npy', nb_pores_800_mean)
                np.save(path + str(nb_im) + '_TTA_mean' + '/' + str(k) + '/' + 'porosity.npy', porosity_800_mean)
                np.save(path + str(nb_im) + '_TTA_mean' + '/' + str(k) + '/' + 'average_width.npy', average_width_800_mean)


def average_ensemble_bio_stats():
    nb_images = [10, 10, 200, 40, 800]
    folders = ['dataset0', 'D', 'D', 'CDEF', 'CDEF']
    name_list = np.arange(1, 26)

    for config in range(5):
        nb_im = nb_images[config]
        folder = folders[config]

        for i in range(0, 5):
            print(i)
            path = './metrics/' + str(folder) + '/'
            if not os.path.exists(path + str(nb_im) + '_ensemble_softfusion_noise15_' + str(i+1)):
                os.makedirs(path + str(nb_im) + '_ensemble_softfusion_noise15_' + str(i+1))
            for k in [6, 7, 9, 10]:
                metrics_0 = np.load(path + '5_' + str(nb_im) + '/' + str(k) + '/surface.npy')
                length = len(metrics_0)
                surface_800_array = np.zeros((5, length))
                nb_pores_800_array = np.zeros((5, length))
                porosity_800_array = np.zeros((5, length))
                average_width_800_array = np.zeros((5, length))

                for j in range(1, 6):
                    surface_800_array[j - 1] = np.load(path + str(name_list[i * 5 + j - 1]) + '_' + str(nb_im) + '_noise15/' + str(k) + '/surface.npy')
                    nb_pores_800_array[j - 1] = np.load(path + str(name_list[i * 5 + j - 1]) + '_' + str(nb_im) + '_noise15/' + str(k) + '/nb_pores.npy')
                    porosity_800_array[j - 1] = np.load(path + str(name_list[i * 5 + j - 1]) + '_' + str(nb_im) + '_noise15/' + str(k) + '/porosity.npy')
                    average_width_800_array[j - 1] = np.load(path + str(name_list[i * 5 + j - 1]) + '_' + str(nb_im) + '_noise15/' + str(k) + '/average_width.npy')

                surface_800_mean = np.mean(surface_800_array, axis=0)
                nb_pores_800_mean = np.mean(nb_pores_800_array, axis=0)
                porosity_800_mean = np.mean(porosity_800_array, axis=0)
                average_width_800_mean = np.mean(average_width_800_array, axis=0)

                if not os.path.exists(path + str(nb_im) + '_ensemble_softfusion_noise15_' + str(i+1) + '/' + str(k)):
                    os.makedirs(path + str(nb_im) + '_ensemble_softfusion_noise15_' + str(i+1) + '/' + str(k))
                np.save(path + str(nb_im) + '_ensemble_softfusion_noise15_' + str(i+1) + '/' + str(k) + '/' + 'surface.npy', surface_800_mean)
                np.save(path + str(nb_im) + '_ensemble_softfusion_noise15_' + str(i+1) + '/' + str(k) + '/' + 'nb_pores.npy', nb_pores_800_mean)
                np.save(path + str(nb_im) + '_ensemble_softfusion_noise15_' + str(i+1) + '/' + str(k) + '/' + 'porosity.npy', porosity_800_mean)
                np.save(path + str(nb_im) + '_ensemble_softfusion_noise15_' + str(i+1) + '/' + str(k) + '/' + 'average_width.npy', average_width_800_mean)


# def average_ensemble_TTA_bio_stats_light():
#     #name_list = [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
#     name_list = np.arange(1, 26) #[5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
#
#     for i in range(0, 5):
#         print(i)
#         path = './metrics/CDEF/'
#         if not os.path.exists(path + '800_ensemble_TTA_softfusion_' + str(i+1)):
#             os.makedirs(path + '800_ensemble_TTA_softfusion_' + str(i+1))
#         for k in [6, 7, 9, 10]:
#             metrics_0 = np.load(path + '5_800/' + str(k) + '/surface.npy')
#             length = len(metrics_0)
#             surface_800_array = np.zeros((5, length))
#             nb_pores_800_array = np.zeros((5, length))
#             porosity_800_array = np.zeros((5, length))
#             average_width_800_array = np.zeros((5, length))
#
#             for j in range(1, 6):
#                 # surface_800_array[j-1] = np.load(path + str(i*5+j) + '_200_wozb/' + str(k) + '/surface.npy')
#                 # nb_pores_800_array[j-1] = np.load(path + str(i*5+j) + '_200_wozb/' + str(k) + '/nb_pores.npy')
#                 # porosity_800_array[j-1] = np.load(path + str(i*5+j) + '_200_wozb/' + str(k) + '/porosity.npy')
#                 # average_width_800_array[j-1] = np.load(path + str(i*5+j) + '_200_wozb/' + str(k) + '/average_width.npy')
#                 surface_800_array[j - 1] = np.load(path + str(name_list[i * 5 + j-1]) + '/800_TTA_mean/' + str(k) + '/surface.npy')
#                 nb_pores_800_array[j - 1] = np.load(path + str(name_list[i * 5 + j-1]) + '/800_TTA_mean/' + str(k) + '/nb_pores.npy')
#                 porosity_800_array[j - 1] = np.load(path + str(name_list[i * 5 + j-1]) + '/800_TTA_mean/' + str(k) + '/porosity.npy')
#                 average_width_800_array[j - 1] = np.load(
#                     path + str(name_list[i * 5 + j-1]) + '/800_TTA_mean/' + str(k) + '/average_width.npy')
#
#             surface_800_mean = np.mean(surface_800_array, axis=0)
#             nb_pores_800_mean = np.mean(nb_pores_800_array, axis=0)
#             porosity_800_mean = np.mean(porosity_800_array, axis=0)
#             average_width_800_mean = np.mean(average_width_800_array, axis=0)
#
#             if not os.path.exists(path + '800_ensemble_TTA_softfusion_' + str(i+1) + '/' + str(k)):
#                 os.makedirs(path + '800_ensemble_TTA_softfusion_' + str(i+1) + '/' + str(k))
#             np.save(path + '800_ensemble_TTA_softfusion_' + str(i+1) + '/' + str(k) + '/' + 'surface.npy', surface_800_mean)
#             np.save(path + '800_ensemble_TTA_softfusion_' + str(i+1) + '/' + str(k) + '/' + 'nb_pores.npy', nb_pores_800_mean)
#             np.save(path + '800_ensemble_TTA_softfusion_' + str(i+1) + '/' + str(k) + '/' + 'porosity.npy', porosity_800_mean)
#             np.save(path + '800_ensemble_TTA_softfusion_' + str(i+1) + '/' + str(k) + '/' + 'average_width.npy', average_width_800_mean)


###############################
# Print functions             #
###############################

def print_mean_std_entries_newtable(root_folder,
                                    list_of_training_sets,
                                    list_of_base_learners,
                                    list_of_images,
                                    metric_name, nb_samples,**kwargs):
    is_light = kwargs.get('light', False)
    is_light2 = kwargs.get('light2', False)
    minmax_indices = kwargs.get('minmax_indices', None)

    metric_array = np.zeros((len(list_of_base_learners), nb_samples*len(list_of_training_sets)))
    is_dsc = (metric_name == 'dsc')
    if not is_dsc:
        metric_array_GT = np.zeros((len(list_of_base_learners), nb_samples*len(list_of_training_sets)))
    for i in range(len(list_of_base_learners)):
        metric_array_i = []
        metric_array_i_GT = []
        for j in range(len(list_of_images)):
            for c in range(len(list_of_training_sets)):
                file_path = root_folder + '/' \
                            + list_of_training_sets[c] + '/' \
                            + list_of_base_learners[i] + '/' \
                            + str(list_of_images[j]) + '/' \
                            + metric_name + '.npy'
                file_path_GT = './metrics/GT/' + str(list_of_images[j]) + '/' + metric_name + '.npy'
                if os.path.isfile(file_path):
                    metric_array_i = np.append(metric_array_i, np.load(file_path))
                    if not is_dsc:
                        metric_GT = np.load(file_path_GT)
                        metric_GT = metric_GT[::4]
                        metric_array_i_GT = np.append(metric_array_i_GT, metric_GT)
                else:
                    return '-'
        metric_array[i] = metric_array_i
        if not is_dsc:
            metric_array_GT[i] = metric_array_i_GT
    if is_dsc:
        #metric_mean = np.mean(metric_array, axis=0)
        #metric_std = np.std(metric_array, axis=0)
        #return str('%0.3f' % np.mean(metric_mean)) + '$\pm$' + str('%0.3f' % np.mean(metric_std))
        metric_mean = np.mean(metric_array, axis=1)
        if is_light:
            return str('%0.3f' % np.mean(metric_mean))
        elif is_light2:
            return str('%0.3f' % np.mean(metric_mean)) + '$\pm$' + str('%0.3f' % np.std(metric_mean))
        else:
            #print('min' + np.where(metric_mean == np.amin(metric_mean)))
            #print('max' + np.where(metric_mean == np.amax(metric_mean)))
            return str('%0.3f' % np.mean(metric_mean)) + '$\pm$' + str('%0.3f' % np.std(metric_mean)) + ' & ' + str('%0.3f' % np.amin(metric_mean)) + ' & ' + str('%0.3f' % np.amax(metric_mean))
    else:
        #metric_mean = np.mean(np.divide(np.absolute(metric_array_GT - metric_array), metric_array_GT), axis=0)*100
        #metric_std = np.divide(np.std(metric_array, axis=0), np.mean(metric_array_GT, axis=0))*100
        #return str('%0.1f\\%%' % np.mean(metric_mean)) + ' $\pm$ ' + str('%0.1f\\%%' % np.mean(metric_std))
        #metric_mean = np.mean(np.divide(np.absolute(metric_array_GT - metric_array), metric_array_GT), axis=1) * 100
        metric_mean = np.mean(np.absolute(metric_array_GT - metric_array), axis=1)
        #ind = np.random.random_integers(10)
        plt.plot(np.absolute(metric_array_GT[2][0:80] - metric_array[2][0:80]))
        plt.savefig('./figure/metric_bias_TTA_2.png')
        print(metric_mean)
        #metric_mean = np.mean(metric_array_GT - metric_array, axis=1)

        metric_GT_mean = np.mean(metric_array_GT, axis=1)

        mn = np.mean(metric_mean) #/ np.mean(metric_GT_mean) * 100
        sd = np.std(metric_mean) #/ np.mean(metric_GT_mean) * 100
        if minmax_indices is not None:
            worst = metric_mean[minmax_indices[0]] #/ np.mean(metric_GT_mean) * 100
            best = metric_mean[minmax_indices[1]] # / np.mean(metric_GT_mean) * 100
        else:
            worst = np.amin(metric_mean) / np.mean(metric_GT_mean) * 100
            best = np.amax(metric_mean) / np.mean(metric_GT_mean) * 100

        if is_light:
            return str('%0.1f' % mn)
        elif is_light2:
            return str('%0.1f' % mn) + '$\pm$' + str('%0.2f' % sd)
        else:
            return str('%0.1f' % mn) + '$\pm$' + str('%0.2f' % sd) + ' & ' + str('%0.1f' % worst) + ' & ' + str('%0.1f' % best)


def print_mean_std_entries_newtable_variance_between_models(root_folder,
                                    list_of_training_sets,
                                    list_of_base_learners,
                                    list_of_images,
                                    metric_name, nb_samples,**kwargs):
    is_light = kwargs.get('light', False)

    metric_array = np.zeros((len(list_of_base_learners), nb_samples*len(list_of_training_sets)))
    is_dsc = (metric_name == 'dsc')
    if not is_dsc:
        metric_array_GT = np.zeros((len(list_of_base_learners), nb_samples*len(list_of_training_sets)))
    for i in range(len(list_of_base_learners)):
        metric_array_i = []
        metric_array_i_GT = []
        for j in range(len(list_of_images)):
            for c in range(len(list_of_training_sets)):
                file_path = root_folder + '/' \
                            + list_of_training_sets[c] + '/' \
                            + list_of_base_learners[i] + '/' \
                            + str(list_of_images[j]) + '/' \
                            + metric_name + '.npy'
                file_path_GT = './metrics/GT/' + str(list_of_images[j]) + '/' + metric_name + '.npy'
                if os.path.isfile(file_path):
                    metric_array_i = np.append(metric_array_i, np.load(file_path))
                    if not is_dsc:
                        metric_GT = np.load(file_path_GT)
                        metric_GT = metric_GT[::4]
                        metric_array_i_GT = np.append(metric_array_i_GT, metric_GT)
                else:
                    print(list_of_base_learners[i])
                    return '-'
        metric_array[i] = metric_array_i
        if not is_dsc:
            metric_array_GT[i] = metric_array_i_GT
    if is_dsc:
        #metric_mean = np.mean(metric_array, axis=0)
        #metric_std = np.std(metric_array, axis=0)
        #return str('%0.3f' % np.mean(metric_mean)) + '$\pm$' + str('%0.3f' % np.mean(metric_std))
        metric_mean = np.mean(metric_array, axis=1)
        if is_light:
            return str('%0.3f' % np.mean(metric_mean))
        else:
            return str('%0.1f' % np.mean(metric_mean)) + ' & ' + str('%0.1f' % np.max(metric_mean)) + ' & ' + str('%0.1f' % np.min(metric_mean)) + ' & ' + str('%0.1f' % np.std(metric_mean))
    else:
        #metric_mean = np.mean(np.divide(np.absolute(metric_array_GT - metric_array), metric_array_GT), axis=0)*100
        #metric_std = np.divide(np.std(metric_array, axis=0), np.mean(metric_array_GT, axis=0))*100
        #if metric_name == 'porosity':
        #    metric_array = metric_array*100

        # E_prediction E_slices
        metric_mean = np.mean(np.absolute(metric_array_GT - metric_array), axis=1)
        #metric_mean_n = np.mean(np.absolute(np.divide(metric_array_GT - metric_array, metric_array_GT))*100, axis=1)
        #metric_mean = np.mean(metric_array_GT - metric_array, axis=1)
        #metric_mean_n = np.divide(np.mean(np.absolute(metric_array_GT - metric_array), axis=1), np.mean(metric_array_GT, axis=1)) * 100

        # E_slices E_predictions
        #metric_mean = np.mean(np.absolute(metric_array_GT - metric_array), axis=0)
        #metric_max = np.amax(np.absolute(metric_array_GT - metric_array), axis=0)
        #metric_min = np.amax(np.absolute(metric_array_GT - metric_array), axis=0)
        #metric_std = np.std(np.absolute(metric_array_GT - metric_array), axis=0)

        # metric_mean_n = np.mean(np.absolute(np.divide(metric_array_GT - metric_array, metric_array_GT)), axis=0)*100
        # metric_max_n = np.amax(np.absolute(np.divide(metric_array_GT - metric_array, metric_array_GT)), axis=0)*100
        # metric_min_n = np.amin(np.absolute(np.divide(metric_array_GT - metric_array, metric_array_GT)), axis=0)*100
        # metric_std_n = np.std(np.absolute(np.divide(metric_array_GT - metric_array, metric_array_GT)), axis=0)*100

        #return str('%0.1f\\%%' % np.mean(metric_mean)) + ' $\pm$ ' + str('%0.1f\\%%' % np.mean(metric_std))
        #metric_mean = np.mean(np.divide(np.absolute(metric_array_GT - metric_array), metric_array_GT), axis=1) * 100
        if is_light:
            return str('%0.1f' % np.mean(metric_std))
            #return str('%0.1f\\%%' % np.mean(metric_std))
        else:
            #if metric_name == 'porosity':
            #    return str('%0.2f' % np.mean(metric_mean)) + ' & ' + str('%0.2f' % np.mean(metric_std))
            #else:
            #return str('%0.1f' % np.mean(metric_mean)) + ' & ' + str('%0.1f' % np.mean(metric_std))
            # mn = np.mean(metric_mean)/np.mean(metric_GT_mean)*100
            # sd = np.mean(metric_std)/np.mean(metric_GT_mean)*100
            # best = np.mean(metric_max)/np.mean(metric_GT_mean)*100
            # worst = np.mean(metric_min)/np.mean(metric_GT_mean)*100
            # max_dev = np.mean(metric_max_dev)/np.mean(metric_GT_mean)*100
            #return str('%0.1f \\%% ' % mn) + ' & ' + str(' [ %0.1f \\%% ' % best) + str(' %0.1f \\%% ]' % worst) + ' & ' + str('%0.1f \\%%' % sd)
            #return str('%0.1f' % mn) + str(' (%0.1f -- ' % best) + str('%0.1f)\\%%' % worst) + ' & ' + str('%0.1f\\%%' % sd)
            #return str('%0.1f' % mn) + ' & ' + str('%0.1f' % max_dev) + ' & ' + str('%0.1f\\%%' % sd)

            # E_prediction E_slices
            if metric_name == 'porosity':
                return str('%0.2f\\%%' % np.mean(metric_mean*100)) + '$\\pm$' + str('%0.2f\\%%' % np.std(metric_mean*100)) + ' & ' + str('%0.1f\\%%' % np.max(metric_mean*100))
            else:
                #return str('%0.1f' % np.mean(metric_mean)) + ' & ' + str('%0.1f' % np.max(metric_mean)) + ' & ' + str('%0.1f' % np.std(metric_mean))
                #return str('%0.3f' % np.mean(metric_mean)) + '$\\pm$' + str('%0.3f' % np.std(metric_mean)) + ' & ' + str('%0.3f' % np.max(metric_mean))
                #return str('%0.0f' % np.mean(metric_mean)) + '$\\pm$' + str('%0.0f' % np.std(metric_mean)) + ' & ' + str('%0.0f' % np.max(metric_mean))
                return str('%0.1f' % np.mean(metric_mean)) + '$\\pm$' + str('%0.1f' % np.std(metric_mean)) + ' & ' + str('%0.1f' % np.max(metric_mean))

                #return str('%0.1f' % np.mean(metric_mean_n)) + ' & ' + str('%0.1f' % np.max(metric_mean_n)) + ' & ' + str(
                #'%0.1f' % np.min(metric_mean_n)) + ' & ' + str('%0.1f' % np.std(metric_mean_n))

            # E_slices E_predictions
            # if metric_name == 'porosity':
            #     return str('%0.3f\\%%' % np.mean(metric_mean * 100)) + ' & ' + str(
            #         '%0.3f\\%%' % np.mean(metric_max_dev * 100)) + ' & ' + str('%0.3f\\%%' % np.mean(metric_std * 100))
            # else:
            #     return str('%0.1f' % np.mean(metric_mean)) + ' & ' + str('%0.1f' % np.mean(metric_max)) + ' & ' + str('%0.1f' % np.mean(metric_min)) + ' & ' + str(
            #         '%0.1f' % np.mean(metric_std))
            # if metric_name == 'porosity':
            #     return str('%0.3f\\%%' % np.mean(metric_mean * 100)) + ' & ' + str(
            #         '%0.3f\\%%' % np.mean(metric_max_dev * 100)) + ' & ' + str('%0.3f\\%%' % np.mean(metric_std * 100))
            # else:
            #     return str('%0.1f\\%%' % np.mean(metric_mean_n)) + ' & ' + str('%0.1f\\%%' % np.mean(metric_max_n)) + ' & ' + str('%0.1f\\%%' % np.mean(metric_min_n)) \
            #            + ' & ' + str('%0.1f\\%%' % np.mean(metric_std_n))


###############################
# Write latex tables          #
###############################

def write_latex_code_ensemble_TTA_not_dsc():
    nb_samples = 84 + 90 + 64 + 52
    metric_name = 'dsc'
    nb_pixels_mask = 0
    for i, value in enumerate([6, 7, 9, 10]):
        mask = np.load('./samples/raw/' + str(value) + '_labels.npy')
        nb_pixels_mask = nb_pixels_mask + np.sum(mask[:, :, ::4])

    print('C(1,10) & ' + print_mean_std_entries_newtable('./metrics',
                                                        ['dataset0'],
                                                        #[str(i) + '_10' for i in
                                                        # np.concatenate((np.arange(1, 12), np.arange(14, 21)))],
                                                        [str(i) + '_10_noise15' for i in range(1, 21)],
                                                        [6, 7, 9, 10],
                                                        metric_name, nb_samples, minmax_indices=[13, 11])
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            ['dataset0'],
                                            ['10_ensemble_hardfusion_noise15'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                      ['dataset0/14'],
                                                      ['10_TTA' + str(i) for i in range(0, 20)],
                                                      [6, 7, 9, 10],
                                                      metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                      ['dataset0/14'],
                                      ['10_TTA_hardfusion'],
                                      [6, 7, 9, 10],
                                      metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    #['dataset0/12'],
                                                    ['dataset0/1'],
                                                    ['10_TTA' + str(i) for i in range(0, 20)],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    #['dataset0/12'],
                                                    ['dataset0/1'],
                                                    ['10_TTA_hardfusion'],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light=True)
          + '\\\\')
    print('C(2,10)  & ' + print_mean_std_entries_newtable('./metrics',
                                                         ['D'],
                                                         [str(i) + '_10_noise15' for i in range(1, 21)],
                                                         [6, 7, 9, 10],
                                                         metric_name, nb_samples, minmax_indices=[14, 10])
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            ['D'],
                                            ['10_ensemble_hardfusion_noise15'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            #['D/15'],
                                            ['D/5'],
                                            ['10_TTA' + str(i) for i in range(0, 20)],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            #['D/15'],
                                                    ['D/5'],
                                            ['10_TTA_hardfusion'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    #['D/11'],
                                                    ['D/12'],
                                                    ['10_TTA' + str(i) for i in range(0, 20)],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    #['D/11'],
                                                    ['D/12'],
                                                    ['10_TTA_hardfusion'],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light=True)
          + '\\\\')
    print('C(2,200)  & ' + print_mean_std_entries_newtable('./metrics',
                                                          ['D'],
                                                          #[str(i) + '_200_wozb' for i in np.concatenate(
                                                          #    (np.arange(5, 10), np.arange(11, 14), np.arange(15, 25)))],
                                                          [str(i) + '_200_noise15' for i in range(1, 21)],
                                                          [6, 7, 9, 10],
                                                          metric_name, nb_samples, minmax_indices=[13, 9])
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            ['D'],
                                            ['200_ensemble_hardfusion_noise15'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            #['D/19'],
                                                    ['D/24'],
                                                    ['200_TTA' + str(i) for i in range(0, 20)],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            #['D/19'],
                                                    ['D/24'],
                                                    ['200_TTA_hardfusion'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    ['D/15'],
                                                    ['200_TTA' + str(i) for i in range(0, 20)],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    ['D/15'],
                                                    ['200_TTA_hardfusion'],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light=True)
          + '\\\\')
    print('C(8,40)  & ' + print_mean_std_entries_newtable('./metrics',
                                                         ['CDEF'],
                                                         [str(i) + '_40_noise15' for i in range(1, 21)],
                                                         [6, 7, 9, 10],
                                                         metric_name, nb_samples, minmax_indices=[11, 6])
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            ['CDEF'],
                                            ['40_ensemble_hardfusion_noise15'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            ['CDEF/12bis'],
                                            ['40_TTA' + str(i) for i in range(0, 20)],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            ['CDEF/12bis'],
                                            ['40_TTA_hardfusion'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    ['CDEF/7bis'],
                                                    ['40_TTA' + str(i) for i in range(0, 20)],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    ['CDEF/7bis'],
                                                    ['40_TTA_hardfusion'],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light=True)
          + '\\\\')
    print('C(8,800)  & ' + print_mean_std_entries_newtable('./metrics',
                                                          ['CDEF'],
                                                          [str(i) + '_800_noise15' for i in range(1, 21)],
                                                          [6, 7, 9, 10],
                                                          metric_name, nb_samples, minmax_indices=[0, 7])
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            ['CDEF'],
                                            ['800_ensemble_hardfusion_noise15'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                            #['CDEF/1'],
                                                    ['CDEF/7bis'],
                                                    ['800_TTA' + str(i) for i in range(0, 20)],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                           # ['CDEF/1'],
                                                    ['CDEF/7bis'],
                                                    ['800_TTA_hardfusion'],
                                            [6, 7, 9, 10],
                                            metric_name, nb_samples, light=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    ['CDEF/6bis'],
                                                    ['800_TTA' + str(i) for i in range(0, 20)],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light2=True)
          + ' & ' + print_mean_std_entries_newtable('./metrics',
                                                    ['CDEF/6bis'],
                                                    ['800_TTA_hardfusion'],
                                                    [6, 7, 9, 10],
                                                    metric_name, nb_samples, light=True)
          + '\\\\')


def write_latex_code_softfusion_table_fusion_ensemble_TTA_both_mean_noise15():
    nb_samples = 84 + 90 + 64 + 52
    metric_name = 'porosity'
    print('C(1,10)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                  ['dataset0'],
                                                                                  # [str(i) + '_40' for i in
                                                                                  #  [1, 6, 11, 16, 21]],
                                                                                  [str(i) + '_10_noise15' for i in
                                                                                   range(1, 26)],
                                                                                  [6, 7, 9, 10],
                                                                                  metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['dataset0'],
                                                                            ['10_ensemble_softfusion_noise15_' + str(i) for i in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['dataset0'],
                                                                            [str(i) + '/10_TTA_mean' for i in
                                                                             range(1, 26)],
                                                                            # [str(i) + '/40_TTA_mean' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['dataset0'],
          #                                                                   ['10_ensemble_TTA_softfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')
    print('C(2,10)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                  ['D'],
                                                                                  # [str(i) + '_40' for i in
                                                                                  #  [1, 6, 11, 16, 21]],
                                                                                  [str(i) + '_10_noise15' for i in
                                                                                   range(1, 26)],
                                                                                  [6, 7, 9, 10],
                                                                                  metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['D'],
                                                                            ['10_ensemble_softfusion_noise15_' + str(i) for i in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['D'],
                                                                            [str(i) + '/10_TTA_mean' for i in
                                                                             range(1, 26)],
                                                                            # [str(i) + '/40_TTA_mean' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['D'],
          #                                                                   ['10_ensemble_TTA_softfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')
    print('C(2,200)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                  ['D'],
                                                                                  # [str(i) + '_40' for i in
                                                                                  #  [1, 6, 11, 16, 21]],
                                                                                  [str(i) + '_200_noise15' for i in range(1, 26)],
                                                                                   #[str(i) + '_200_wozb' for i in range(20,30)],
                                                                                   [6, 7, 9, 10],
                                                                                  metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['D'],
                                                                            ['200_ensemble_softfusion_noise15_' + str(i) for i in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['D'],
                                                                            [str(i) + '/200_TTA_mean' for i in
                                                                             np.concatenate((np.arange(5, 10), np.arange(11, 31)))],
                                                                            #[str(i) + '/200_TTA_mean' for i in
                                                                            # range(20,30)],
                                                                            # [str(i) + '/40_TTA_mean' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['D'],
          #                                                                   ['200_ensemble_TTA_softfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')
    print('C(8,40)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                 ['CDEF'],
                                                                                 # [str(i) + '_40' for i in
                                                                                 #  [1, 6, 11, 16, 21]],
                                                                                  [str(i) + '_40_noise15' for i in
                                                                                   range(1, 26)],
                                                                                 [6, 7, 9, 10],
                                                                                 metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['CDEF'],
                                                                            ['40_ensemble_softfusion_noise15_' + str(i) for i in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['CDEF'],
                                                                            [str(i) + '/40_TTA_mean' for i in range(1, 26)],
                                                                            #[str(i) + '/40_TTA_mean' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['CDEF'],
          #                                                                   ['40_ensemble_TTA_softfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')
    print('C(8,800)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                  ['CDEF'],
                                                                                  #[str(i) + '_800' for i in
                                                                                  # [1, 6, 11, 16, 21]],
                                                                                   [str(i) + '_800_noise15' for i in
                                                                                    range(1, 26)],
                                                                                  [6, 7, 9, 10],
                                                                                  metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['CDEF'],
                                                                            ['800_ensemble_softfusion_noise15_' + str(i) for i
                                                                             in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['CDEF'],
                                                                            [str(i) + '/800_TTA_mean' for i in range(1, 26)],
                                                                            #[str(i) + '/800_TTA_mean' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['CDEF'],
          #                                                                   ['800_ensemble_TTA_softfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')


def write_latex_code_softfusion_table_fusion_ensemble_TTA_both_mv_noise15():
    nb_samples = 84 + 90 + 64 + 52
    metric_name = 'porosity'
    print('C(1,10)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                  ['dataset0'],
                                                                                  # [str(i) + '_40' for i in
                                                                                  #  [1, 6, 11, 16, 21]],
                                                                                  [str(i) + '_10_noise15' for i in
                                                                                   range(1, 26)],
                                                                                  [6, 7, 9, 10],
                                                                                  metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['dataset0'],
                                                                            ['10_ensemble_hardfusion_' + str(i)  + '_noise15' for i in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['dataset0'],
                                                                            [str(i) + '/10_TTA_hardfusion' for i in
                                                                             range(1, 26)],
                                                                            # [str(i) + '/40_TTA_hardfusion' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['dataset0'],
          #                                                                   ['10_ensemble_TTA_hardfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')
    print('C(2,10)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                  ['D'],
                                                                                  # [str(i) + '_40' for i in
                                                                                  #  [1, 6, 11, 16, 21]],
                                                                                  [str(i) + '_10_noise15' for i in
                                                                                   range(1, 26)],
                                                                                  [6, 7, 9, 10],
                                                                                  metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['D'],
                                                                            ['10_ensemble_hardfusion_' + str(i) + '_noise15' for i in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['D'],
                                                                            [str(i) + '/10_TTA_hardfusion' for i in
                                                                             range(1, 26)],
                                                                            # [str(i) + '/40_TTA_hardfusion' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['D'],
          #                                                                   ['10_ensemble_TTA_hardfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')
    print('C(2,200)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                  ['D'],
                                                                                  # [str(i) + '_40' for i in
                                                                                  #  [1, 6, 11, 16, 21]],
                                                                                  [str(i) + '_200_noise15' for i in range(1, 26)],
                                                                                   #[str(i) + '_200_wozb' for i in range(20,30)],
                                                                                   [6, 7, 9, 10],
                                                                                  metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['D'],
                                                                            ['200_ensemble_hardfusion_' + str(i) + '_noise15' for i in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['D'],
                                                                            [str(i) + '/200_TTA_hardfusion' for i in
                                                                             np.concatenate((np.arange(5, 10), np.arange(11, 31)))],
                                                                            # [str(i) + '/200_TTA_hardfusion' for i in
                                                                            #  range(20,30)],
                                                                            # [str(i) + '/40_TTA_hardfusion' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['D'],
          #                                                                   ['200_ensemble_TTA_hardfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')
    print('C(8,40)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                 ['CDEF'],
                                                                                 # [str(i) + '_40' for i in
                                                                                 #  [1, 6, 11, 16, 21]],
                                                                                  [str(i) + '_40_noise15' for i in
                                                                                   range(1, 26)],
                                                                                 [6, 7, 9, 10],
                                                                                 metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['CDEF'],
                                                                            ['40_ensemble_hardfusion_' + str(i) + '_noise15' for i in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['CDEF'],
                                                                             [str(i) + '/40_TTA_hardfusion' for i in
                                                                             range(1, 26)],
                                                                            #[str(i) + '/40_TTA_hardfusion' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['CDEF'],
          #                                                                   ['40_ensemble_TTA_hardfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')
    print('C(8,800)  & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                                  ['CDEF'],
                                                                                  #[str(i) + '_800' for i in
                                                                                  # [1, 6, 11, 16, 21]],
                                                                                   [str(i) + '_800_noise15' for i in
                                                                                    range(1, 26)],
                                                                                  [6, 7, 9, 10],
                                                                                  metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['CDEF'],
                                                                            ['800_ensemble_hardfusion_' + str(i) + '_noise15' for i
                                                                             in
                                                                             range(1, 6)],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
                                                                            ['CDEF'],
                                                                             [str(i) + '/800_TTA_hardfusion' for i in
                                                                             range(1, 26)],
                                                                            #[str(i) + '/800_TTA_hardfusion' for i in
                                                                            # [1, 6, 11, 16, 20]],
                                                                            [6, 7, 9, 10],
                                                                            metric_name, nb_samples)
          # + ' & ' + print_mean_std_entries_newtable_variance_between_models('./metrics',
          #                                                                   ['CDEF'],
          #                                                                   ['800_ensemble_TTA_hardfusion_' + str(i) for
          #                                                                    i in
          #                                                                    range(1, 6)],
          #                                                                   [6, 7, 9, 10],
          #                                                                   metric_name, nb_samples)
          + '\\\\')


################################
# Plot figures                 #
################################

def build_scatter_plot2():
    surface_model_8_800_ensemble = np.zeros(25)
    surface_model_8_40_ensemble = np.zeros(25)
    surface_model_8_800_tta = np.zeros(80)
    surface_model_8_40_tta = np.zeros(80)
    nb_pores_model_8_800_ensemble = np.zeros(25)
    nb_pores_model_8_40_ensemble = np.zeros(25)
    nb_pores_model_8_800_tta = np.zeros(80)
    nb_pores_model_8_40_tta = np.zeros(80)

    for k in range(0, 25):
        for i in [6, 7, 9, 10]:
            surface_model_8_800_ensemble[k] = surface_model_8_800_ensemble[k] + np.mean(
                np.load('./metrics/CDEF/' + str(k + 1) + '_800_noise15/' + str(i) + '/surface.npy'))
            surface_model_8_40_ensemble[k] = surface_model_8_40_ensemble[k] + np.mean(
                np.load('./metrics/CDEF/' + str(k + 1) + '_40_noise15/' + str(i) + '/surface.npy'))
            nb_pores_model_8_800_ensemble[k] = nb_pores_model_8_800_ensemble[k] + np.mean(
                np.load('./metrics/CDEF/' + str(k + 1) + '_800_noise15/' + str(i) + '/nb_pores.npy'))
            nb_pores_model_8_40_ensemble[k] = nb_pores_model_8_40_ensemble[k] + np.mean(
                np.load('./metrics/CDEF/' + str(k + 1) + '_40_noise15/' + str(i) + '/nb_pores.npy'))
    for k, value in enumerate([0, 2, 3, 4]):
        for j in range(0, 20):
            for i in [6, 7, 9, 10]:
                surface_model_8_800_tta[k*20+j] = surface_model_8_800_tta[k*20+j] + np.mean(
                    np.load('./metrics/CDEF/' + str(value + 1) + '/800_TTA' + str(j) + '/' + str(i) + '/surface.npy'))
                surface_model_8_40_tta[k*20+j] = surface_model_8_40_tta[k*20+j] + np.mean(
                    np.load('./metrics/CDEF/' + str(value + 1) + '/40_TTA' + str(j) + '/' + str(i) + '/surface.npy'))
                nb_pores_model_8_800_tta[k*20+j] = nb_pores_model_8_800_tta[k*20+j] + np.mean(
                    np.load('./metrics/CDEF/' + str(value + 1) + '/800_TTA' + str(j) + '/' + str(i) + '/nb_pores.npy'))
                nb_pores_model_8_40_tta[k*20+j] = nb_pores_model_8_40_tta[k*20+j] + np.mean(
                    np.load('./metrics/CDEF/' + str(value + 1) + '/40_TTA' + str(j) + '/' + str(i) + '/nb_pores.npy'))
    init = ['Initializations different from 1, 2, 3, 4' for i in range(25)]
    init[0] = 'Initialization 1'
    init[2] = 'Initialization 2'
    init[3] = 'Initialization 3'
    init[4] = 'Initialization 4'
    d_800 = {'Base learners': np.concatenate((['Initializations 1 to 25, no TTA' for i in range(0, 25)],
                                      ['Initialization 1, different TTA' for i in range(0, 20)],
                                      ['Initialization 2, different TTA' for i in range(0, 20)],
                                      ['Initialization 3, different TTA' for i in range(0, 20)],
                                      ['Initialization 4, different TTA' for i in range(0, 20)])),
            'Initialization': np.concatenate((init,
                                             ['Initialization 1' for i in range(0, 20)],
                                             ['Initialization 2' for i in range(0, 20)],
                                             ['Initialization 3' for i in range(0, 20)],
                                             ['Initialization 4' for i in range(0, 20)])),
            'Area': np.concatenate((surface_model_8_800_ensemble/4, surface_model_8_800_tta/4)),
            'Number of pores': np.concatenate((nb_pores_model_8_800_ensemble/4, nb_pores_model_8_800_tta/4))}
    df_800 = pd.DataFrame(data=d_800)
    d_40 = {'Base learners': np.concatenate((['Initializations 1 to 25, no TTA' for i in range(0, 25)],
                                             ['Initialization 1, different TTA' for i in range(0, 20)],
                                             ['Initialization 2, different TTA' for i in range(0, 20)],
                                             ['Initialization 3, different TTA' for i in range(0, 20)],
                                             ['Initialization 4, different TTA' for i in range(0, 20)])),
            'Initialization': np.concatenate((init,
                                              ['Initialization 1' for i in range(0, 20)],
                                              ['Initialization 2' for i in range(0, 20)],
                                              ['Initialization 3' for i in range(0, 20)],
                                              ['Initialization 4' for i in range(0, 20)])),
            'Area': np.concatenate((surface_model_8_40_ensemble / 4, surface_model_8_40_tta / 4)),
            'Number of pores': np.concatenate((nb_pores_model_8_40_ensemble / 4, nb_pores_model_8_40_tta / 4))}
    df_40 = pd.DataFrame(data=d_40)
    fig = plt.figure(figsize=(9, 5))
    fig.add_axes([0.1, 0.15, 0.5, 0.80])
    sns.scatterplot(x="Area", y='Number of pores', hue="Base learners", style='Initialization', palette="deep", data=df_40, legend='brief')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim([70000, 78000])
    plt.ylim([320, 350])
    plt.grid()
    plt.savefig('./figure/scatterplot_surface_model_8_40_ensemble_noise15.png')
    plt.close()

    fig = plt.figure(figsize=(9, 5))
    fig.add_axes([0.1, 0.15, 0.5, 0.80])
    sns.scatterplot(x="Area", y='Number of pores', hue="Base learners", style='Initialization', palette="deep",
                    data=df_800, legend='brief')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim([70000, 82000])
    plt.ylim([310, 370])
    plt.grid()
    plt.savefig('./figure/scatterplot_surface_model_8_800_ensemble_noise15.png')
    plt.close()


def correlation_DSC_entropy():
    # folders_name = ['./metrics/CDEF/40_ensemble_hardfusion/6/',
    #                 './metrics/CDEF/40_ensemble_hardfusion/7/',
    #                 './metrics/CDEF/40_ensemble_hardfusion/9/',
    #                 './metrics/CDEF/40_ensemble_hardfusion/10/']
    # folders_name = ['./metrics/CDEF/800_ensemble_hardfusion/',
    #                 './metrics/CDEF/40_ensemble_hardfusion/',
    #                 './metrics/D/200_ensemble_hardfusion/',
    #                 './metrics/D/10_ensemble_hardfusion/',
    #                 './metrics/dataset0/10_ensemble_hardfusion/']
    # sets_name = ['C(8,800)', 'C(8,40)', 'C(2,200)', 'C(2,10)', 'C(1,10)']
    folders_name = ['./metrics/CDEF/800_ensemble_hardfusion_noise15/',
                    './metrics/CDEF/40_ensemble_hardfusion_noise15/',
                    './metrics/D/200_ensemble_hardfusion_noise15/',
                    './metrics/D/10_ensemble_hardfusion_noise15/',
                    './metrics/dataset0/10_ensemble_hardfusion_noise15/',
                    './metrics/CDEF/800_TTA_hardfusion/',
                    './metrics/CDEF/40_TTA_hardfusion/',
                    './metrics/D/200_TTA_hardfusion/',
                    './metrics/D/10_TTA_hardfusion/',
                    './metrics/dataset0/10_TTA_hardfusion/'
                    ]
    sets_name = ['Ens. C(8,800)', 'Ens. C(8,40)', 'Ens. C(2,200)', 'Ens. C(2,10)', 'Ens. C(1,10)',
                 'TTA C(8,800)', 'TTA C(8,40)', 'TTA C(2,200)', 'TTA C(2,10)', 'TTA C(1,10)']

    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle('Correlation between DSC and entropy level', fontsize=16)

    for k, value in enumerate(folders_name):
        dsc_array = np.zeros(84 + 90 + 64 + 52)
        entropy_array = np.zeros(84 + 90 + 64 + 52)
        metric_array = np.zeros(84 + 90 + 64 + 52)
        counter = 0
        for i, value_i in enumerate([6, 7, 9, 10]):
            dsc = np.load(value + str(value_i) + '/' + 'dsc.npy')
            # metric_name = 'surface'
            # metric = np.load(value + str(value_i) + '/' + metric_name + '.npy')
            # metric_GT = np.load('./metrics/GT/' + str(value_i) + '/' + metric_name + '.npy')
            # metric_diff = np.absolute(metric-metric_GT[::4])
            entropy_norm_area = np.load(value + str(value_i) + '/' + 'entropy_norm_area.npy')
            dsc_array[counter:counter + len(dsc)] = np.log(np.divide(dsc, 1-dsc))
            entropy_array[counter:counter + len(entropy_norm_area)] = entropy_norm_area
            #metric_array[counter:counter + len(metric_diff)] = metric_diff
            counter = counter + len(dsc)

        r_input = np.zeros((2, 84 + 90 + 64 + 52))
        r_input[0] = dsc_array
        r_input[1] = entropy_array
        r = np.corrcoef(r_input)
        # r_input2 = np.zeros((2, 84 + 90 + 64 + 52))
        # r_input2[0] = metric_array
        # r_input2[1] = entropy_array
        # r2 = np.corrcoef(r_input2)
        # print(r)

        # With seaborn
        d = {'Logit(DSC)': dsc_array,
             'Entropy level': entropy_array}
        df = pd.DataFrame(data=d)
        sns.scatterplot(ax=axes[int(k / 5), k % 5], data=df, x='Logit(DSC)', y='Entropy level')
        sns.regplot(ax=axes[int(k / 5), k % 5], x='Logit(DSC)', y='Entropy level', data=df)
        axes[int(k / 5), k % 5].annotate('r = ' + str(np.round(r[0, 1], decimals=3)), xy=(0, 0), xytext=(2, 0.9))
        axes[int(k / 5), k % 5].set_title(sets_name[k], fontsize=12)
        axes[int(k / 5), k % 5].set_xlim([0, 4])
        axes[int(k / 5), k % 5].set_ylim([0, 1])
        axes[int(k / 5), k % 5].grid()

        # d = {'Area': metric_array,
        #      'Entropy level': entropy_array}
        # df = pd.DataFrame(data=d)
        # sns.scatterplot(ax=axes[int(k / 5), k % 5], data=df, x='Area', y='Entropy level')
        # sns.regplot(ax=axes[int(k / 5), k % 5], x='Area', y='Entropy level', data=df)
        # axes[int(k / 5), k % 5].annotate('r = ' + str(np.round(r2[0, 1], decimals=3)), xy=(0, 0), xytext=(2, 0.9))
        # axes[int(k / 5), k % 5].set_title(sets_name[k], fontsize=12)
        # #axes[int(k / 5), k % 5].set_xlim([0, 4])
        # #axes[int(k / 5), k % 5].set_ylim([0, 1])
        # axes[int(k / 5), k % 5].grid()

    for ax in axes.flat:
        ax.label_outer()

    #for i, ax in enumerate(axes.flat):
    #    ax.set_title(titles[i])

    plt.savefig('./metrics/figures/correlation_dsc_entropy_noise15.png')
    #plt.savefig('./metrics/figures/correlation_' + metric_name + '_entropy.png')
    plt.close()


def plot_bio_stat_choose_models_1_3(metric_name, y_label, title, y_scaling_factor, y_lim, folders_name, sets_name, **kwargs):
    is_dsc = kwargs.get('is_dsc', False)
    # Loop on images
    for i, value in enumerate([6, 7, 9, 10]):
        # Load GT metrics and build GT data frame
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for k in range(len(metric_name)):
            if is_dsc is False:
                folder_name = './metrics/GT/' + str(value) + '/'
                metric_GT = np.load(folder_name + metric_name[k] + '.npy')
                metric_GT = metric_GT[::4]
                nb_slices_GT = len(metric_GT)
                slices_GT = np.arange(nb_slices_GT)

                d_GT = {'Slices indices': slices_GT, y_label[k]: metric_GT * y_scaling_factor[k]}
                df_GT = pd.DataFrame(data=d_GT)

            list_of_folders = folders_name

            metric_list = []
            slices_list = []

            # Loop on models
            for j, name_j in enumerate(list_of_folders):
                folder_name = name_j + str(value) + '/'
                metric_list = metric_list + (np.load(folder_name + metric_name[k] + '.npy')).tolist()
                nb_slices = len(np.load(folder_name + metric_name[k] + '.npy'))
                #slices_list = slices_list + (np.arange(nb_slices) * 4).tolist()
                slices_list = slices_list + (np.arange(nb_slices)).tolist()

            # Build models data frames
            #fig.suptitle(title[k] + str(value), fontsize=16)

            d = {'Slices indices': slices_list,
                  y_label[k]: np.asarray(metric_list) * y_scaling_factor[k]}
            df = pd.DataFrame(data=d)
            sns.lineplot(ax=axes[k], x='Slices indices', y=y_label[k], data=df, ci='sd', label='DL')
            if is_dsc is False:
                sns.lineplot(ax=axes[k], x='Slices indices', y=y_label[k], data=df_GT, ci=None, label='Manual')
            axes[k].set_title(sets_name[k], fontsize=12)
            axes[k].set_ylim(y_lim[k])
            axes[k].grid()

        fig.savefig('./metrics/figures/' + str(value) + '_all_metrics.png')
        plt.close()


def run_plot_bio_stat_choose_models_1_3():  # separate figures for different images, compatible with all folders
    folders_name = ['./metrics/CDEF/' + str(i) + '_800_noise15/' for i in range(1, 26)]
    sets_name = ['Area', 'Number of pores', 'Porosity']
    metric_name = ['surface', 'nb_pores', 'porosity']
    ylim = [[0, 150], [0, 700], [0, 25]]
    ylabel = ['Cartilage area [kpixels]', 'Number of pores', 'Cartilage porosity [%]']
    scaling_factor = [1/1000, 1, 100]

    plot_bio_stat_choose_models_1_3(metric_name, ylabel, 'None', scaling_factor, ylim,
                                folders_name, sets_name)


############################
# Fusion                   #
############################

def fusion(list_of_dirs, nb_epochs, value):
    #mask_init = np.load('./results/' + list_of_dirs[0] + '/firstval600/predictions/' + str(value)
    #                    + '_prediction_epoch' + str(nb_epochs) + '_std.npy')
    #mask_init = np.load('/DATA/jeaneliott/cartilage_results/' + list_of_dirs[0] + '/predictions/' + str(value) + '_prediction_fusion_TTA_new_20predictions.npy')
    mask_init = np.load('/DATA/jeaneliott/cartilage_results/' + list_of_dirs[0] + '/predictions/' + str(value) + '_prediction_epoch' + str(nb_epochs) + '_std_noise15.npy')
    #mask_init = np.load('./results/' + list_of_dirs[0] + '/firstval600/predictions/' + str(value) + '_prediction_fusion_TTA_new_20predictions.npy')

    predictions_acc = np.zeros(mask_init.shape)
    for j, filename in enumerate(list_of_dirs):
        #mask = np.load('./results/' + filename + '/firstval600/predictions/' + str(value) + '_prediction_epoch' + str(nb_epochs) + '_std.npy')
        mask = np.load('/DATA/jeaneliott/cartilage_results/' + filename + '/predictions/' + str(value) + '_prediction_epoch' + str(nb_epochs) + '_std_noise15.npy')
        predictions_acc = predictions_acc + mask

        # mask = np.load('/DATA/jeaneliott/cartilage_results/' + filename + '/predictions/' + str(value) + '_prediction_fusion_TTA_new_20predictions.npy')
        # #mask = np.load('./results/' + filename + '/firstval600/predictions/' + str(value) + '_prediction_fusion_TTA_new_20predictions.npy')
        # mask_thr = np.zeros(mask.shape)
        # mask_thr[mask > 10] = 1
        # predictions_acc = predictions_acc + mask_thr

    return predictions_acc


def fusion2(list_of_dirs, value):
    mask_init = np.load('./results/' + list_of_dirs[0] + '/firstval600/predictions/' + str(value)
                        + '_prediction_fusion_TTA_new_20predictions.npy')
    predictions_acc = np.zeros(mask_init.shape)
    for j, filename in enumerate(list_of_dirs):
        mask = np.load('./results/' + filename + '/firstval600/predictions/' + str(value)
                       + '_prediction_fusion_TTA_new_20predictions.npy')
        mask_thr = np.zeros(mask.shape)
        mask_thr[mask > 10] = 1
        predictions_acc = predictions_acc + mask_thr
    return predictions_acc


def save_fusion_ensemble_full():
    fusion_dico = {
            '800': [
            'model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            'model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'],

            '40': [
           'model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
           'model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'],

            '200_d': [
               'model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
               'model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'],

              '10_d': [
                   'model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_2d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_3d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_4d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_10d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'],

              '10_0': [
                   'model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_2dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_3dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_4dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_5dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_6dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_7dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_8dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_9dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_10dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_11dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_12dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_13dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_14dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_15dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_16dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_17dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_18dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_19dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                   'model_unet_2d_modality_real5layers_partition_0_200_20dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10']
    }
    epochs = [112, 2240, 450, 9000, 9000]
    folder_names = ['CDEF', 'CDEF', 'D', 'D', 'dataset0']
    compute_metric_en = 1
    nTrain = [800, 40, 200, 10, 10]

    for i, key in enumerate(fusion_dico.keys()):
        list_of_dirs = fusion_dico[key]
        for j, value in enumerate([6, 7, 9, 10]):
            im_fusion = fusion(list_of_dirs, epochs[i], value)
            #im_fusion = fusion2(list_of_dirs, value)
            np.save('./results_fusion/' + folder_names[i] + '/' + str(value) + '_prediction_fusion_' + key + '_ensemble_noise15.npy', im_fusion.astype(np.uint8))
            if compute_metric_en:
                image = np.load('./samples/raw/' + str(value) + '_image.npy')
                image = image[:, :, ::4]
                im_fusion_thr = np.zeros(im_fusion.shape)
                im_fusion_thr[im_fusion > 10] = 1
                mask = np.transpose(im_fusion_thr, (1, 2, 0))
                folder_name = './metrics/' + folder_names[i] + '/' + str(nTrain[i]) + '_ensemble_hardfusion_noise15/' + str(value) + '/'
                #compute_bio_stats(image, mask, folder_name)
                mask_labels = np.load('./samples/raw/' + str(value) + '_labels.npy')
                if i == 7:
                    mask_labels[:, :, 79] = mask_labels[:, :, 78]
                    mask_labels[:, :, 80] = mask_labels[:, :, 81]
                mask_labels = mask_labels[:, :, ::4]
                compute_bio_stats(image, mask, folder_name, mask_3d_labels=mask_labels)


def save_fusion_ensemble(ensemble_number):
    if ensemble_number is 1:
        fusion_dico = {
            '800': [
                'model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'],

            '40': [
                'model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'],

            '200_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'],

            '10_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_2d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_3d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_4d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'],

            '10_0': [
                'model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_2dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_3dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_4dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_5dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10']
        }
    elif ensemble_number is 2:
        fusion_dico = {
            '800': [
                'model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'],

            '40': [
                'model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'],

            '200_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'],

            '10_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_10d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'],

            '10_0': [
                'model_unet_2d_modality_real5layers_partition_0_200_6dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_7dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_8dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_9dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_10dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10']
        }
    elif ensemble_number is 3:
        fusion_dico = {
            '800': [
                'model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'],

            '40': [
                'model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'],

            '200_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'],

            '10_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'],

            '10_0': [
                'model_unet_2d_modality_real5layers_partition_0_200_11dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_12dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_13dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_14dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_15dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10']
        }
    elif ensemble_number is 4:
        fusion_dico = {
            '800': [
                'model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'],

            '40': [
                'model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'],

            '200_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'],

            '10_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'],

            '10_0': [
                'model_unet_2d_modality_real5layers_partition_0_200_16dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_17dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_18dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_19dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_20dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10']
        }
    elif ensemble_number is 5:
        fusion_dico = {
            '800': [
                'model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                'model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'
            ],

            '40': [
                'model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                'model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'
            ],

            '200_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_26d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_27d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_28d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_29d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                'model_unet_2d_modality_real5layers_partition_0_200_30d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'
            ],

            '10_d': [
                'model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'
            ],

            '10_0': [
                'model_unet_2d_modality_real5layers_partition_0_200_21dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_22dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_23dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_24dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                'model_unet_2d_modality_real5layers_partition_0_200_25dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'
            ]
        }

    epochs = [112, 2240, 450, 9000, 9000]
    folder_names = ['CDEF', 'CDEF', 'D', 'D', 'dataset0']
    compute_metric_en = 1
    nTrain = [800, 40, 200, 10, 10]

    for i, key in enumerate(fusion_dico.keys()):
        list_of_dirs = fusion_dico[key]
        for j, value in enumerate([6, 7, 9, 10]):
            im_fusion = fusion(list_of_dirs, epochs[i], value)
            #im_fusion = fusion2(list_of_dirs, value)
            np.save('./results_fusion/' + folder_names[i] + '/' + str(value) + '_prediction_fusion_' + key + '_ensemble_5inits_' + str(ensemble_number) + '.npy', im_fusion.astype(np.uint8))
            if compute_metric_en:
                image = np.load('./samples/raw/' + str(value) + '_image.npy')
                image = image[:, :, ::4]
                im_fusion_thr = np.zeros(im_fusion.shape)
                im_fusion_thr[im_fusion>10] = 1
                mask = np.transpose(im_fusion_thr, (1, 2, 0))
                folder_name = './metrics/' + folder_names[i] + '/' + str(nTrain[i]) + '_ensemble_hardfusion_' + str(ensemble_number) + '_noise15/' + str(value) + '/'
                print(image.shape)
                print(mask.shape)
                #compute_bio_stats(image, mask, folder_name)
                mask_labels = np.load('./samples/raw/' + str(value) + '_labels.npy')
                if i == 7:
                    mask_labels[:, :, 79] = mask_labels[:, :, 78]
                    mask_labels[:, :, 80] = mask_labels[:, :, 81]
                mask_labels = mask_labels[:, :, ::4]
                compute_bio_stats(image, mask, folder_name, mask_3d_labels=mask_labels)


##############################
# Entropy computation        #
##############################

def compute_slicewise_epistemic_entropy(list_of_dirs, nb_epochs, folder):
    for i, value in enumerate([6, 7, 9, 10]):
        im_fusion = fusion(list_of_dirs, nb_epochs, value)
        sh = im_fusion.shape
        entropy_array = np.zeros(sh[0])
        p = im_fusion / len(list_of_dirs)
        print(np.amax(p))
        epsilon = 0.0000000001
        entropy = -np.multiply(p, np.log2(p + epsilon)) - np.multiply((1 - p), np.log2(1 - p + epsilon))
        print(np.amax(entropy))
        for j in range(sh[0]):
            entropy_array[j] = np.sum(entropy[j])
        folder_name = './metrics/' + folder + str(value) + '/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        np.save(folder_name + 'entropy.npy', entropy_array)
        area = np.load(folder_name + 'surface.npy')
        entropy_norm = np.divide(entropy_array, area)
        np.save(folder_name + 'entropy_norm_area.npy', entropy_norm)
        entropy_sqrt_norm = np.divide(entropy_array, np.sqrt(area))
        print(entropy_sqrt_norm)
        np.save(folder_name + 'entropy_norm_sqrt_area.npy', entropy_sqrt_norm)


def compute_slicewise_aleatoric_entropy(dir, nb_predictions, folder):
    for i, value in enumerate([6, 7, 9, 10]):
        im_fusion = np.load(dir + str(value) + '_prediction_fusion_TTA_new_' + str(nb_predictions) + 'predictions.npy')
        sh = im_fusion.shape
        entropy_array = np.zeros(sh[0])
        p = im_fusion / nb_predictions
        print(np.amax(p))
        epsilon = 0.0000000001
        entropy = -np.multiply(p, np.log2(p + epsilon)) - np.multiply((1 - p), np.log2(1 - p + epsilon))
        print(np.amax(entropy))
        for j in range(sh[0]):
            entropy_array[j] = np.sum(entropy[j])
        folder_name = './metrics/' + folder + str(value) + '/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        np.save(folder_name + 'entropy.npy', entropy_array)
        area = np.load(folder_name + 'surface.npy')
        entropy_norm = np.divide(entropy_array, area)
        np.save(folder_name + 'entropy_norm_area.npy', entropy_norm)
        entropy_sqrt_norm = np.divide(entropy_array, np.sqrt(area))
        print(entropy_sqrt_norm)
        np.save(folder_name + 'entropy_norm_sqrt_area.npy', entropy_sqrt_norm)


def build_epistemic_curve_2():
    #x_axis = [5, 10, 15, 20]
    x_axis = [20]
    fusion_dico = {'800': [ 'model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
                            'model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'
                            ],
                   '40': [
                       'model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
                       'model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'
                   ],
                   '200_d': [   'model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                                'model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'

                                ],
                   '10_d': ['model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_2d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_3d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_4d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_10d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                                'model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'
                               ],
                   '10_0': ['model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_2dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_3dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_4dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_5dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_6dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_7dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_8dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_9dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_10dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_11dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_12dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_13dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_14dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_15dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_16dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_17dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_18dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_19dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                               'model_unet_2d_modality_real5layers_partition_0_200_20dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'
                              ]}
    epochs = [112, 2240, 450, 9000, 9000]
    folder = ['CDEF/800_ensemble_hardfusion_noise15/', 'CDEF/40_ensemble_hardfusion_noise15/', 'D/200_ensemble_hardfusion_noise15/', 'D/10_ensemble_hardfusion_noise15/', 'dataset0/10_ensemble_hardfusion_noise15/']
    entropy_array = np.zeros((len(fusion_dico), len(x_axis)))
    for i, key in enumerate(fusion_dico.keys()):
        for j, value in enumerate(x_axis):
            list_of_dirs = fusion_dico[key]
            #entropy_array[i, j] = compute_epistemic_entropy(list_of_dirs[0:int(value)], epochs[i])
            compute_slicewise_epistemic_entropy(list_of_dirs[0:int(value)], epochs[i], folder[i])
    #np.save('./metrics/entropy/epistemic_uncertainty_2_20inits.npy', entropy_array)


def build_aleatoric_curve():
    #x_axis = [5, 10, 15, 20]
    x_axis = [20]
    list_of_dirs = ['./results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/',
                    './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/firstval600/predictions/',
                    './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/firstval600/predictions/',
                    './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
                    './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/'
                    ]
    folder = ['CDEF/800_TTA_hardfusion/', 'CDEF/40_TTA_hardfusion/', 'D/200_TTA_hardfusion/', 'D/10_TTA_hardfusion/', 'dataset0/10_TTA_hardfusion/']

    entropy_array = np.zeros((len(list_of_dirs), len(x_axis)))
    for i in range(len(list_of_dirs)):
        for j, value in enumerate(x_axis):
            #entropy_array[i, j] = compute_aleatoric_entropy(list_of_dirs[i], value)
            print(j)
            compute_slicewise_aleatoric_entropy(list_of_dirs[i], value, folder[i])
    #print(entropy_array)
    #np.save('./metrics/entropy/aleatoric_uncertainty_20inits.npy', entropy_array)


#########################
# Mutual information    #
#########################

def compute_mutual_information_ensemble(labels, predictions, thr):
    labels = labels[:, :, ::4]
    p_labels = np.sum(labels)/labels.size
    #print(p_labels)
    #print(entropy([p_labels, 1 - p_labels]))
    predictions = np.transpose(predictions, (1, 2, 0))
    predictions_mv = np.zeros(predictions.shape)
    predictions_mv[predictions>thr] = 1
    #return normalized_mutual_info_score(np.reshape(labels, (1, labels.size))[0], np.reshape(predictions, (1, predictions.size))[0])
    return mutual_info_score(np.reshape(labels, (1, labels.size))[0], np.reshape(predictions, (1, predictions.size))[0])/entropy([p_labels, 1-p_labels])
    #return mutual_info_score(np.reshape(labels, (1, labels.size))[0], np.reshape(predictions_mv, (1, predictions_mv.size))[0])/entropy([p_labels, 1-p_labels])
    #return mutual_info_score(np.reshape(labels, (1, labels.size))[0], np.reshape(predictions, (1, predictions.size))[0])


def run_mi_computation():
    folders_names = {'20pred': ['./results_fusion/CDEF/',
                     './results_fusion/CDEF/',
                     './results_fusion/D/',
                     './results_fusion/D/',
                     './results_fusion/dataset0/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/firstval600/predictions/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/firstval600/predictions/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/'
                     ],
                     '5pred': ['./results_fusion/CDEF/',
                                './results_fusion/CDEF/',
                                './results_fusion/D/',
                                './results_fusion/D/',
                                './results_fusion/dataset0/',
                                './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/',
                                './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/firstval600/predictions/',
                                './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/firstval600/predictions/',
                                './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
                                './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/'
                                ],
                     # '1pred': [       './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
                     #           './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/'
                     #           ],
                     '1pred': [
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/predictions/',
                         '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/predictions/'
                         ]
                     }
    fusion_names = {'20pred': ['_prediction_fusion_800_ensemble_noise15.npy',
                    '_prediction_fusion_40_ensemble_noise15.npy',
                    '_prediction_fusion_200_d_ensemble_noise15.npy',
                    '_prediction_fusion_10_d_ensemble_noise15.npy',
                    '_prediction_fusion_10_0_ensemble_noise15.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy'
                    ],
                '5pred': ['_prediction_fusion_800_ensemble_5inits.npy',
                    '_prediction_fusion_40_ensemble_5inits.npy',
                    '_prediction_fusion_200_d_ensemble_5inits.npy',
                    '_prediction_fusion_10_d_ensemble_5inits.npy',
                    '_prediction_fusion_10_0_ensemble_5inits.npy',
                    '_prediction_fusion_TTA_new_5predictions.npy',
                    '_prediction_fusion_TTA_new_5predictions.npy',
                    '_prediction_fusion_TTA_new_5predictions.npy',
                    '_prediction_fusion_TTA_new_5predictions.npy',
                    '_prediction_fusion_TTA_new_5predictions.npy'
                    ],
                '1pred': ['_prediction_epoch112_std_noise15.npy',
                    '_prediction_epoch2240_std_noise15.npy',
                    '_prediction_epoch450_std_noise15.npy',
                    '_prediction_epoch9000_std_noise15.npy',
                    '_prediction_epoch9000_std_noise15.npy',
                    '_prediction_epoch112_std_noise15.npy',
                    '_prediction_epoch2240_std_noise15.npy',
                    '_prediction_epoch450_std_noise15.npy',
                    '_prediction_epoch9000_std_noise15.npy',
                    '_prediction_epoch9000_std_noise15.npy'
                    ]}
    mi_table = np.zeros((3, 10))
    thr_table = [10, 2, 0.5]
    for k, key_k in enumerate(folders_names.keys()):
        list_of_folders_names = folders_names[key_k]
        list_of_fusion_names = fusion_names[key_k]
        for i, key in enumerate(list_of_folders_names):
            print(i)
            mi_acc = 0
            for j, value in enumerate([6, 7, 9, 10]):
                mi_acc = mi_acc + compute_mutual_information_ensemble(np.load('./samples/raw/' + str(value) + '_labels.npy'),
                                                                    np.load(key + str(value) + list_of_fusion_names[i]), thr_table[k])
            mi_acc = mi_acc/4
            print(mi_acc)
            mi_table[k, i] = mi_acc
    np.save('./metrics/mi/mi_table_my_normalization_paper_noise15.npy', mi_table)


def write_latex_code_mi():
    mi = np.load('./metrics/mi/mi_table_my_normalization_paper_noise15.npy')
    print('1 & 10  & ' + str('%0.3f' % mi[0, 4])
          + ' & ' + str('%0.3f' % mi[1, 4])
          + ' & ' + str('%0.3f' % mi[2, 4])
          + ' & ' + str('%0.3f' % mi[0, 9])
          + ' & ' + str('%0.3f' % mi[1, 9])
          + ' & ' + str('%0.3f' % mi[2, 9])
          + '\\\\')
    print('\hline')
    print('2 & 10  & ' + str('%0.3f' % mi[0, 3])
          + ' & ' + str('%0.3f' % mi[1, 3])
          + ' & ' + str('%0.3f' % mi[2, 3])
          + ' & ' + str('%0.3f' % mi[0, 8])
          + ' & ' + str('%0.3f' % mi[1, 8])
          + ' & ' + str('%0.3f' % mi[2, 8])
          + '\\\\')
    print('\hline')
    print('2 & 200  & ' + str('%0.3f' % mi[0, 2])
          + ' & ' + str('%0.3f' % mi[1, 2])
          + ' & ' + str('%0.3f' % mi[2, 2])
          + ' & ' + str('%0.3f' % mi[0, 7])
          + ' & ' + str('%0.3f' % mi[1, 7])
          + ' & ' + str('%0.3f' % mi[2, 7])
          + '\\\\')
    print('\hline')
    print('8 & 40  & ' + str('%0.3f' % mi[0, 1])
          + ' & ' + str('%0.3f' % mi[1, 1])
          + ' & ' + str('%0.3f' % mi[2, 1])
          + ' & ' + str('%0.3f' % mi[0, 6])
          + ' & ' + str('%0.3f' % mi[1, 6])
          + ' & ' + str('%0.3f' % mi[2, 6])
          + '\\\\')
    print('\hline')
    print('8 & 800  & ' + str('%0.3f' % mi[0, 0])
          + ' & ' + str('%0.3f' % mi[1, 0])
          + ' & ' + str('%0.3f' % mi[2, 0])
          + ' & ' + str('%0.3f' % mi[0, 5])
          + ' & ' + str('%0.3f' % mi[1, 5])
          + ' & ' + str('%0.3f' % mi[2, 5])
          + '\\\\')
    print('\hline')


##########################
# Figures                #
##########################

def save_figures_entropy_uncert_seg():
    folders_names = ['./results_fusion/dataset0/',
                     './results_fusion/D/',
                     './results_fusion/D/',
                     './results_fusion/CDEF/',
                     './results_fusion/CDEF/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/firstval600/predictions/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/firstval600/predictions/',
                     './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/',
                     ]

    fusion_names = [
        '_prediction_fusion_10_0_ensemble_noise15.npy',
        '_prediction_fusion_10_d_ensemble_noise15.npy',
        '_prediction_fusion_200_d_ensemble_noise15.npy',
        '_prediction_fusion_40_ensemble_noise15.npy',
        '_prediction_fusion_800_ensemble_noise15.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy',
                    '_prediction_fusion_TTA_new_20predictions.npy'
                    ]

    titles = ['MV', 'Entropy', 'Proba. map',
              'MV', 'Entropy', 'Proba. map']
    config = ['C(1,10)', 'C(2,10)', 'C(2,200)', 'C(8,40)', 'C(8,800)']
    for j, value in enumerate([6, 7, 9, 10]):
        labels = np.load('./samples/raw/' + str(value) + '_labels.npy')
        labels = labels[:, :, ::4]
        image = np.load('./samples/raw/' + str(value) + '_image.npy')
        image = image[:, :, ::4]
        sh = labels.shape

        for k in range(0, sh[-1], 10):
            # Initialize figure
            fig, axs = plt.subplots(5, 6, figsize=(12, 10))
            fig.subplots_adjust(hspace=.05, wspace=.05)

            # Plot labels
            labels_k = labels[:, :, k]
            image_k = image[:, :, k]
            selem = disk(5)
            labels_eroded = erosion(labels_k, selem)
            labels_contour = labels_k - labels_eroded
            labels_contour_image = draw_color_single(image_k.astype(np.double), labels_contour, color_channel=0, alpha=1)

            for i, key in enumerate(folders_names):
                predictions = np.load(key + str(value) + fusion_names[i])
                predictions_k = predictions[k, :, :]
                hardfusion = np.zeros(predictions_k.shape)
                hardfusion[predictions_k>10] = 1
                hardfusion_eroded = erosion(hardfusion, selem)
                hardfusion_contour = hardfusion - hardfusion_eroded
                hardfusion_contour_image = draw_color_two(image_k.astype(np.double),labels_contour, hardfusion_contour, color_channel=0, color_channel2=2,
                                                      alpha=1)
                p = predictions_k / 20
                epsilon = 0.00001
                entropy = -np.multiply(p, np.log2(p + epsilon)) - np.multiply((1 - p), np.log2(1 - p + epsilon))
                entropy = entropy * 255
                entropy_mask = np.zeros(entropy.shape)
                entropy_mask[p == 1] = 1
                entropy_map = draw_color_single_line(entropy.astype(np.double), labels_contour, entropy_mask,
                                                           color_channel=0,
                                                           alpha=1)
                p = p*255
                probability_map = draw_color_single_line(p.astype(np.double), labels_contour, entropy_mask, color_channel=0,
                                                            alpha=1)

                if i < 5:
                    axs[i % 5, 0].imshow(hardfusion_contour_image)
                    axs[i % 5, 0].set_ylabel(config[i])
                    axs[i % 5, 0].set_xlabel('Ensembling')
                    axs[i % 5, 1].imshow(entropy)
                    axs[i % 5, 1].set_xlabel('Ensembling')
                    axs[i % 5, 2].imshow(probability_map)
                    axs[i % 5, 2].set_xlabel('Ensembling')
                else:
                    axs[i % 5, 3].imshow(hardfusion_contour_image)
                    axs[i % 5, 3].set_xlabel('TTA')
                    axs[i % 5, 4].imshow(entropy)
                    axs[i % 5, 4].set_xlabel('TTA')
                    axs[i % 5, 5].imshow(probability_map)
                    axs[i % 5, 5].set_xlabel('TTA')

            for ax in axs.flat:
                ax.label_outer()
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            for i, ax in enumerate(axs.flat[0:6]):
                ax.set_title(titles[i])
            plt.savefig(
                './figure/fusion/entropy/comparison/' + str(value) + '/entropy_proba_seg_slice_' + str(k) + '_noise15.png')
            plt.close()


def save_figure_intro():
    predictions_im6 = np.load('./results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/6_prediction_epoch112_std.npy')
    predictions_im9 = np.load('./results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/9_prediction_epoch112_std.npy')
    predictions_im7 = np.load('./results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/7_prediction_epoch112_std.npy')
    predictions_im6_2 = np.load(
        './results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/6_prediction_epoch112_std.npy')
    predictions_im9_2 = np.load(
        './results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/9_prediction_epoch112_std.npy')
    predictions_im7_2 = np.load(
        './results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/7_prediction_epoch112_std.npy')

    selem = disk(5)

    predictions1 = predictions_im7[20, :, :]
    predictions1_eroded = erosion(predictions1, selem)
    predictions1_contour = predictions1 - predictions1_eroded
    predictions2 = predictions_im9[20, :, :]
    predictions2_eroded = erosion(predictions2, selem)
    predictions2_contour = predictions2 - predictions2_eroded
    predictions3 = predictions_im6[50, :, :]
    predictions3_eroded = erosion(predictions3, selem)
    predictions3_contour = predictions3 - predictions3_eroded
    predictions1_2 = predictions_im7_2[20, :, :]
    predictions1_eroded_2 = erosion(predictions1_2, selem)
    predictions1_contour_2 = predictions1_2 - predictions1_eroded_2
    predictions2_2 = predictions_im9_2[20, :, :]
    predictions2_eroded_2 = erosion(predictions2_2, selem)
    predictions2_contour_2 = predictions2_2 - predictions2_eroded_2
    predictions3_2 = predictions_im6_2[50, :, :]
    predictions3_eroded_2 = erosion(predictions3_2, selem)
    predictions3_contour_2 = predictions3_2 - predictions3_eroded_2

    labels1 = np.load('./samples/raw/7_labels.npy')
    labels1 = labels1[:, :, 80]
    labels1_eroded = erosion(labels1, selem)
    labels1_contour = labels1 - labels1_eroded
    image1 = np.load('./samples/raw/7_image.npy')
    image1 = image1[:, :, 80]
    labels2 = np.load('./samples/raw/9_labels.npy')
    labels2 = labels2[:, :, 80]
    labels2_eroded = erosion(labels2, selem)
    labels2_contour = labels2 - labels2_eroded
    image2 = np.load('./samples/raw/9_image.npy')
    image2 = image2[:, :, 80]
    labels3 = np.load('./samples/raw/6_labels.npy')
    labels3 = labels3[:, :, 200]
    labels3_eroded = erosion(labels3, selem)
    labels3_contour = labels3 - labels3_eroded
    image3 = np.load('./samples/raw/6_image.npy')
    image3 = image3[:, :, 200]

    contour_image1 = draw_color_two(image1.astype(np.double), labels1_contour, predictions1_contour, color_channel=0, color_channel2=2, alpha=1)
    contour_image2 = draw_color_two(image2.astype(np.double), labels2_contour, predictions2_contour, color_channel=0, color_channel2=2, alpha=1)
    contour_image3 = draw_color_two(image3.astype(np.double), labels3_contour, predictions3_contour, color_channel=0, color_channel2=2, alpha=1)
    contour_image1_2 = draw_color_two(image1.astype(np.double), labels1_contour, predictions1_contour_2,
                                       color_channel=0, color_channel2=2, alpha=1)
    contour_image2_2 = draw_color_two(image2.astype(np.double), labels2_contour, predictions2_contour_2,
                                       color_channel=0, color_channel2=2, alpha=1)
    contour_image3_2 = draw_color_two(image3.astype(np.double), labels3_contour, predictions3_contour_2,
                                       color_channel=0, color_channel2=2, alpha=1)

    fig, axs = plt.subplots(3, 3, figsize=(7.5, 7.5))
    fig.subplots_adjust(hspace=.05, wspace=.05)
    axs[0, 0].imshow(image1, cmap='gray')
    axs[0, 1].imshow(image2, cmap='gray')
    axs[0, 2].imshow(image3, cmap='gray')
    axs[1, 0].imshow(contour_image1)
    axs[1, 0].set_ylabel('Initialization 1')
    axs[1, 1].imshow(contour_image2)
    axs[1, 2].imshow(contour_image3)
    axs[2, 0].imshow(contour_image1_2)
    axs[2, 0].set_ylabel('Initialization 2')
    axs[2, 1].imshow(contour_image2_2)
    axs[2, 2].imshow(contour_image3_2)

    for ax in axs.ravel():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.savefig(
        './figure/fusion/entropy/comparison/figure_intro.png')
    plt.close()


######################
# Calibration        #
######################

def compute_posterior_ece(labels_names, fusion_names, save_name):
    acc_joined_labels_fusion = np.zeros((2, 21))
    for i, value in enumerate(fusion_names):
        labels = labels_names[i]
        fusion = fusion_names[i]
        labels = labels[:, :, ::4]
        fusion = np.transpose(fusion, (1, 2, 0))
        # fusion_thr = np.zeros(fusion.shape)
        # fusion_thr[fusion>10] = 1
        # fusion = fusion_thr
        for index, label_index in np.ndenumerate(labels):
            acc_joined_labels_fusion[int(label_index), int(fusion[index])] = acc_joined_labels_fusion[int(label_index), int(fusion[index])] + 1
    path = './metrics/calibration/'
    #np.save(path + save_name + 'hard_fusion_acc_joined_labels_fusion.npy', acc_joined_labels_fusion)
    np.save(path + save_name + '_acc_joined_labels_fusion.npy', acc_joined_labels_fusion)


def call_compute_posterior_ece():
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load('./results_fusion/CDEF/6_prediction_fusion_40_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/CDEF/7_prediction_fusion_40_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/CDEF/9_prediction_fusion_40_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/CDEF/10_prediction_fusion_40_ensemble_noise15.npy')],
    #                       '40_ensemble_noise15')
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load('./results_fusion/CDEF/6_prediction_fusion_800_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/CDEF/7_prediction_fusion_800_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/CDEF/9_prediction_fusion_800_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/CDEF/10_prediction_fusion_800_ensemble_noise15.npy')],
    #                       '800_ensemble_noise15')
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load('./results_fusion/D/6_prediction_fusion_200_d_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/D/7_prediction_fusion_200_d_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/D/9_prediction_fusion_200_d_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/D/10_prediction_fusion_200_d_ensemble_noise15.npy')],
    #                       '200_d_ensemble_noise15')
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load('./results_fusion/D/6_prediction_fusion_10_d_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/D/7_prediction_fusion_10_d_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/D/9_prediction_fusion_10_d_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/D/10_prediction_fusion_10_d_ensemble_noise15.npy')],
    #                       '10_d_ensemble_noise15')
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load('./results_fusion/dataset0/6_prediction_fusion_10_0_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/dataset0/7_prediction_fusion_10_0_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/dataset0/9_prediction_fusion_10_0_ensemble_noise15.npy'),
    #                        np.load('./results_fusion/dataset0/10_prediction_fusion_10_0_ensemble_noise15.npy')],
    #                       '10_dataset0_ensemble_noise15')

    # TTA_paths = ['./results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/',
    #                      './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/firstval600/predictions/',
    #                      './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/firstval600/predictions/',
    #                      './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
    #                      './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/'
    #                     ]
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load(TTA_paths[1] + '6_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[1] + '7_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[1] + '9_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[1] + '10_prediction_fusion_TTA_new_20predictions.npy')],
    #                       '40_TTA')
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load(TTA_paths[0] + '6_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[0] + '7_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[0] + '9_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[0] + '10_prediction_fusion_TTA_new_20predictions.npy')],
    #                       '800_TTA')
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load(TTA_paths[2] + '6_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[2] + '7_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[2] + '9_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[2] + '10_prediction_fusion_TTA_new_20predictions.npy')],
    #                       '200_d_TTA')
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load(TTA_paths[3] + '6_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[3] + '7_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[3] + '9_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[3] + '10_prediction_fusion_TTA_new_20predictions.npy')],
    #                       '10_d_TTA')
    # compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
    #                        np.load('./samples/raw/7_labels.npy'),
    #                        np.load('./samples/raw/9_labels.npy'),
    #                        np.load('./samples/raw/10_labels.npy')],
    #                       [np.load(TTA_paths[4] + '6_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[4] + '7_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[4] + '9_prediction_fusion_TTA_new_20predictions.npy'),
    #                        np.load(TTA_paths[4] + '10_prediction_fusion_TTA_new_20predictions.npy')],
    #                       '10_dataset0_TTA')
    # TTA_paths = ['./results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/firstval600/predictions/',
    #                      './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/firstval600/predictions/',
    #                      './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/firstval600/predictions/',
    #                      './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/',
    #                      './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/firstval600/predictions/'
    #                     ]
    TTA_paths = [
        '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800/predictions/',
        '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40/predictions/',
        '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200/predictions/',
        '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/predictions/',
        '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10/predictions/'
        ]
    compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
                           np.load('./samples/raw/7_labels.npy'),
                           np.load('./samples/raw/9_labels.npy'),
                           np.load('./samples/raw/10_labels.npy')],
                          [np.load(TTA_paths[1] + '6_prediction_epoch2240_std_noise15.npy'),
                           np.load(TTA_paths[1] + '7_prediction_epoch2240_std_noise15.npy'),
                           np.load(TTA_paths[1] + '9_prediction_epoch2240_std_noise15.npy'),
                           np.load(TTA_paths[1] + '10_prediction_epoch2240_std_noise15.npy')],
                          '40_single_noise15')
    compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
                           np.load('./samples/raw/7_labels.npy'),
                           np.load('./samples/raw/9_labels.npy'),
                           np.load('./samples/raw/10_labels.npy')],
                          [np.load(TTA_paths[0] + '6_prediction_epoch112_std_noise15.npy'),
                           np.load(TTA_paths[0] + '7_prediction_epoch112_std_noise15.npy'),
                           np.load(TTA_paths[0] + '9_prediction_epoch112_std_noise15.npy'),
                           np.load(TTA_paths[0] + '10_prediction_epoch112_std_noise15.npy')],
                          '800_single_noise15')
    compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
                           np.load('./samples/raw/7_labels.npy'),
                           np.load('./samples/raw/9_labels.npy'),
                           np.load('./samples/raw/10_labels.npy')],
                          [np.load(TTA_paths[2] + '6_prediction_epoch450_std_noise15.npy'),
                           np.load(TTA_paths[2] + '7_prediction_epoch450_std_noise15.npy'),
                           np.load(TTA_paths[2] + '9_prediction_epoch450_std_noise15.npy'),
                           np.load(TTA_paths[2] + '10_prediction_epoch450_std_noise15.npy')],
                          '200_d_single_noise15')
    compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
                           np.load('./samples/raw/7_labels.npy'),
                           np.load('./samples/raw/9_labels.npy'),
                           np.load('./samples/raw/10_labels.npy')],
                          [np.load(TTA_paths[3] + '6_prediction_epoch9000_std_noise15.npy'),
                           np.load(TTA_paths[3] + '7_prediction_epoch9000_std_noise15.npy'),
                           np.load(TTA_paths[3] + '9_prediction_epoch9000_std_noise15.npy'),
                           np.load(TTA_paths[3] + '10_prediction_epoch9000_std_noise15.npy')],
                          '10_d_single_noise15')
    compute_posterior_ece([np.load('./samples/raw/6_labels.npy'),
                           np.load('./samples/raw/7_labels.npy'),
                           np.load('./samples/raw/9_labels.npy'),
                           np.load('./samples/raw/10_labels.npy')],
                          [np.load(TTA_paths[4] + '6_prediction_epoch9000_std_noise15.npy'),
                           np.load(TTA_paths[4] + '7_prediction_epoch9000_std_noise15.npy'),
                           np.load(TTA_paths[4] + '9_prediction_epoch9000_std_noise15.npy'),
                           np.load(TTA_paths[4] + '10_prediction_epoch9000_std_noise15.npy')],
                          '10_dataset0_single_noise15')


def compute_calibration_metrics():
    joint_acc_names = ['./metrics/calibration/800_single_noise15_acc_joined_labels_fusion.npy',
                            './metrics/calibration/40_single_noise15_acc_joined_labels_fusion.npy',
                           './metrics/calibration/200_d_single_noise15_acc_joined_labels_fusion.npy',
                           './metrics/calibration/10_d_single_noise15_acc_joined_labels_fusion.npy',
                           './metrics/calibration/10_dataset0_single_noise15_acc_joined_labels_fusion.npy',
                            './metrics/calibration/800_ensemble_noise15_acc_joined_labels_fusion.npy',
                            './metrics/calibration/40_ensemble_noise15_acc_joined_labels_fusion.npy',
                           './metrics/calibration/200_d_ensemble_noise15_acc_joined_labels_fusion.npy',
                           './metrics/calibration/10_d_ensemble_noise15_acc_joined_labels_fusion.npy',
                           './metrics/calibration/10_dataset0_ensemble_noise15_acc_joined_labels_fusion.npy',
                           './metrics/calibration/800_TTA_acc_joined_labels_fusion.npy',
                           './metrics/calibration/40_TTA_acc_joined_labels_fusion.npy',
                           './metrics/calibration/200_d_TTA_acc_joined_labels_fusion.npy',
                           './metrics/calibration/10_d_TTA_acc_joined_labels_fusion.npy',
                           './metrics/calibration/10_dataset0_TTA_acc_joined_labels_fusion.npy']
    # joint_acc_names = ['./metrics/calibration/800_single_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/40_single_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/200_d_single_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/10_d_single_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/10_dataset0_single_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/800_ensemblehard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/40_ensemblehard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/200_d_ensemblehard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/10_d_ensemblehard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/10_dataset0_ensemblehard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/800_TTAhard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/40_TTAhard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/200_d_TTAhard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/10_d_TTAhard_fusion_acc_joined_labels_fusion.npy',
    #                    './metrics/calibration/10_dataset0_TTAhard_fusion_acc_joined_labels_fusion.npy']
    short_names = ['800_single',
                 '40_single',
                 '200_d_single',
                 '10_d_single',
                 '10_dataset0_single',
                 '800_ensemble',
                 '40_ensemble',
                 '200_d_ensemble',
                 '10_d_ensemble',
                 '10_dataset0_ensemble',
                 '800_TTA',
                 '40_TTA',
                 '200_d_TTA',
                 '10_d_TTA',
                 '10_dataset0_TTA']

    cal = np.zeros((5, 3 * 3))
    for i, value in enumerate(joint_acc_names):
        joint_acc = np.load(value)
        nb_tot = np.sum(joint_acc)
        ece_table = np.zeros(21)
        brier_table = np.zeros(21)
        nll_table = np.zeros(21)
        for j in range(21):
            nb_j = np.sum(joint_acc[:, j])
            ece_table[j] = nb_j/nb_tot * np.absolute(joint_acc[1, j] / max(nb_j, 1) - j/20) *100
            brier_table[j] = joint_acc[0, j]*np.square(j/20) + joint_acc[1, j]*np.square(j/20-1)
            espilon = 0.0000000001
            nll_table[j] = -joint_acc[0, j]*np.log2(1-j/20+espilon)-joint_acc[1, j]*np.log2(j/20+espilon)
        ece = np.sum(ece_table)
        brier_score = np.sum(brier_table)/nb_tot
        nll_table = np.sum(nll_table)/nb_tot
        cal[i % 5, int(i/5)*3] = ece
        cal[i % 5, int(i/5)*3+1] = brier_score
        cal[i % 5, int(i/5)*3+2] = nll_table
    np.save('./metrics/calibration/cal_noise15.npy', cal)


def write_calibration_table():
    cal = np.load('./metrics/calibration/cal_noise15.npy')
    names = ['C(8,800)', 'C(8,40)', 'C(2,200)', 'C(2,10)', 'C(1,10)']
    for j in reversed(range(5)):
        print(names[j] + ' & ' +
              str('%0.3f' % cal[j, 0]) + ' & ' +
              str('%0.3f' % cal[j, 1]) + ' & ' +
              str('%0.3f' % cal[j, 2]) + ' & ' +
              str('%0.3f' % cal[j, 3]) + ' & ' +
              str('%0.3f' % cal[j, 4]) + ' & ' +
              str('%0.3f' % cal[j, 5]) + ' & ' +
              str('%0.3f' % cal[j, 6]) + ' & ' +
              str('%0.3f' % cal[j, 7]) + ' & ' +
              str('%0.3f' % cal[j, 8]) + '\\\\')
