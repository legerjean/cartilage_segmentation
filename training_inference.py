from core.utils_core import *
#from core.train import *
#from core.predict import *
import numpy as np
import csv
import pickle
from scipy.ndimage import *
from scipy.ndimage.interpolation import rotate
from skimage import feature
from skimage.morphology import erosion
import random
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import time
from core.online_data_augmentation_new import *
import matplotlib.gridspec as gridspec
import pickle
from tensorflow import random
from functions import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_segmenter_2d(name, gpu, nb_epochs, nTrain, images_range):
    parameters = {'nTrain':             nTrain,
                  'nVal':               4*100,
                  'nTest':              0,
                  'model':              'unet_2d',
                  'n_layers':           5,
                  'n_feat_maps':        16,
                  'batch_size':         5,
                  'nb_epochs':          nb_epochs,
                  'lr':                 1e-4,
                  'loss':               'dice_loss_2d',
                  'wd':                 0,
                  'dropout':            0,
                  'bn':                 1,
                  'en_online':          1,
                  'init':               'glorot_normal',
                  'modality':           name}
    nTrainMax = 8*100
    parameters_entries = ('model', 'modality', 'wd', 'bn', 'nb_epochs', 'init', 'modality', 'n_layers', 'nTrain')
    cv_train_model(parameters, parameters_entries, data_dir='./samples/partitions/', image_size=[1024, 1024], gpu=gpu,
                   images_file='image_partition_0_200.npy',
                   masks_file='mask_partition_0_200.npy',
                   nTrainMax=nTrainMax,
                   images_range=images_range)


def predict_segmenter_2d(option, en_TTA, **kwargs):
    if option == 1:
        my_src_dir_list = ['./results/model_unet_2d_modality_real5layers_partition_0_200_1c_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                        './results/model_unet_2d_modality_real5layers_partition_0_200_1e_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
                       './results/model_unet_2d_modality_real5layers_partition_0_200_1f_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'
                        ]
        epoch_to_load = 9000
        letter = ['C', 'E', 'F']
        compute_metrics_en = 1
        nTrain = 10
    elif option == 2:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset1_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset4_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset8_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'
            ]
        epoch_to_load = 9000
        letter = ['dataset1', 'dataset4', 'dataset8']
        compute_metrics_en = 1
        nTrain = 10
    elif option == 3:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_6c_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6e_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6f_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'
            ]
        epoch_to_load = 450
        letter = ['C', 'E', 'F']
        compute_metrics_en = 1
        nTrain = 200
    elif option == 4:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'
        ]
        epoch_to_load = 112
        letter = ['CDEF']
        compute_metrics_en = 1
        nTrain = 800
    elif option == 5:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'
           ]
        epoch_to_load = 112
        letter = ['CDEF']
        compute_metrics_en = 0
        nTrain = 800
    elif option == 6:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10'

           ]
        epoch_to_load = 9000
        letter = ['D']
        compute_metrics_en = 0
        nTrain = 10
    elif option == 7:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_26d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
        ]
        epoch_to_load = 450
        letter = ['D']
        compute_metrics_en = 0
        nTrain = 200
    elif option == 8:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_21dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',

           ]
        epoch_to_load = 9000
        letter = ['dataset0']
        compute_metrics_en = 0
        nTrain = 10
    elif option == 9:
        my_src_dir_list = [
           './results/model_unet_2d_modality_real5layers_partition_0_200_26d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_27d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_28d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_29d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'
            # './results/model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #      './results/model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800'
            ]
        save_dir = [ '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_26d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                     '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_27d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                     '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_28d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
                     '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_29d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200'
        ]

        # my_src_dir_list = [
        #             './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #             './results/model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #             './results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #             './results/model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #             './results/model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'
        #     # './results/model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     # './results/model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     # './results/model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     # './results/model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     # './results/model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     # './results/model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        #     './results/model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40'
        #]
        epoch_to_load = 9000
        letter = [#'CDEF/2', 'CDEF/3', 'CDEF/4', 'CDEF/5', 'CDEF/6','CDEF/7',
                #'CDEF/8', 'CDEF/9',
                'CDEF/1_synchro', 'CDEF/2_synchro', 'CDEF/3_synchro', 'CDEF/4_synchro', 'CDEF/5_synchro'
                  #'CDEF/11', 'CDEF/12', 'CDEF/13', 'CDEF/14', 'CDEF/15', 'CDEF/16',
            #'CDEF/17', 'CDEF/18', 'CDEF/19', 'CDEF/20'
                  ]
        compute_metrics_en = 1
        nTrain = 200
    elif option == 10:
        my_src_dir_list = [
            # './results/model_unet_2d_modality_real5layers_partition_0_200_2d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_3d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_4d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_10d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_30d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',

            # './results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',

        ]
        save_dir = [
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_2d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_3d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_4d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_10d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
             '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_30d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',

            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',

        ]
        epoch_to_load = 9000
        letter = ['D/6', 'D/7']
        compute_metrics_en = 1
        nTrain = 200
    elif option == 11:
        my_src_dir_list = [
            # './results/model_unet_2d_modality_real5layers_partition_0_200_2dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_3dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_4dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_5dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_6dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_7dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_8dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_9dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_10dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_11dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_12dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_13dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_14dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_15dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_16dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_17dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_18dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_19dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_20dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_21dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_22dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_23dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_24dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_25dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_21dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_22dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_23dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_24dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_25dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',

        ]
        save_dir = [
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_2dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_3dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_4dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_5dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_10dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_16dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_17dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_18dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            #'/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_19dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_20dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_24dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_25dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',

        ]
        epoch_to_load = 9000
        letter = [#'dataset0/11', 'dataset0/13', 'dataset0/15', 'dataset0/16', 'dataset0/17', 'dataset0/18', 'dataset0/19',
                  'dataset0/20', 'dataset0/21', 'dataset0/22', 'dataset0/23', 'dataset0/24', 'dataset0/25']
        compute_metrics_en = 1
        nTrain = 10
    elif option == 12:
        my_src_dir_list = [
            # './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_32d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',

        ]
        save_dir = [
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           #  '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           #  '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           #  '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_32d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',

            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',

        ]
        epoch_to_load = 450
        letter = [#'D/6', 'D/7', 'D/8', 'D/9', 'D/11',
                  'D/32'
        ]
        compute_metrics_en = 1
        nTrain = 200
    elif option == 13:
        my_src_dir_list = [
            # './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        ]
        save_dir = [
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6bis_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',

        ]
        epoch_to_load = 112
        letter = [#'D/6', 'D/7', 'D/8', 'D/9', 'D/11',
                  'CDEF/6bis'
        ]
        compute_metrics_en = 1
        nTrain = 800
    elif option == 14:
        my_src_dir_list = [
            # './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # './results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # './results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            #'./results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',

        ]
        save_dir = [
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
           # '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',

        ]
        epoch_to_load = 2240
        letter = [#'D/6', 'D/7', 'D/8', 'D/9', 'D/11',
                  'D/15'
        ]
        compute_metrics_en = 1
        nTrain = 200
    elif option == 15:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_2dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_3dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_4dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_5dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_7dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_8dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_9dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_10dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_11dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_12dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_13dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_14dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_15dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_16dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_17dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_18dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_19dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_20dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_21dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_22dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_23dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_24dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_25dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
        ]
        save_dir = [
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_2dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_3dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_4dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_5dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_10dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_16dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_17dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_18dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_19dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_20dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_24dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_25dataset0_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
        ]
        nTrain = 10
        epoch_to_load = 9000
        letter = ['dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0',
                  'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0', 'dataset0']
        compute_metrics_en = 1
    elif option == 16:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_2d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_3d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_4d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_10d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            './results/model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
        ]
        save_dir = [
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_2d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_3d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_4d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_10d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_9000_init_glorot_normal_n_layers_5_nTrain_10',
        ]
        nTrain = 10
        epoch_to_load = 9000
        letter = ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D',
                  'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
        compute_metrics_en = 1
    elif option == 17:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            './results/model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        ]
        save_dir = [
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_2240_init_glorot_normal_n_layers_5_nTrain_40',
        ]
        nTrain = 40
        epoch_to_load = 2240
        letter = ['CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF',
                  'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF',
                  'CDEF']
        compute_metrics_en = 1
    elif option == 18:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            './results/model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        ]
        save_dir = [
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_1_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_2_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_3_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_4_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_5_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_10_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_16_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_17_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_18_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_19_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_20_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_24_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_25_brightness_noise_wd_0_bn_1_nb_epochs_112_init_glorot_normal_n_layers_5_nTrain_800',
        ]
        nTrain = 800
        epoch_to_load = 112
        letter = ['CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF',
                  'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF', 'CDEF']
        compute_metrics_en = 1
    elif option == 19:
        my_src_dir_list = [
            './results/model_unet_2d_modality_real5layers_partition_0_200_dddd_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_26d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_27d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_28d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_29d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            './results/model_unet_2d_modality_real5layers_partition_0_200_30d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
        ]
        save_dir = [
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_5d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_6d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_7d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_8d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_9d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_11d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_12d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_13d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_14d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_15d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_16d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_17d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_18d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_19d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_20d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_21d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_22d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_23d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_24d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_25d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_26d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_27d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_28d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_29d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
            '/DATA/jeaneliott/cartilage_results/model_unet_2d_modality_real5layers_partition_0_200_30d_brightness_noise_wd_0_bn_1_nb_epochs_450_init_glorot_normal_n_layers_5_nTrain_200',
        ]
        nTrain = 200
        epoch_to_load = 450
        letter = ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D',
                  'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
        compute_metrics_en = 1
    if en_TTA:
        for k, value in enumerate(my_src_dir_list):
            my_src_dir = value
            for i in range(0, 20):
                tda_angle = 180 * np.random.uniform(-1, 1)
                intensity = np.random.uniform(-40, 40)
                noise_std = 15
                cv_predict_model(my_src_dir, en_TTA, data_dir='./samples/raw/', gpu=2, tda_angle=tda_angle,
                                 epoch_to_load=epoch_to_load, intensity=intensity, noise_std=noise_std, index=i,
                                 save_dir=save_dir[k])
                for j in [6, 7, 9, 10]:
                    # src_dir = save_dir[i] + '/firstval600'
                    src_dir = save_dir[k]
                    pred = np.load(src_dir + '/predictions/' + str(j) + '_prediction_TTA' + str(i) + '.npy')
                    if compute_metrics_en:
                        image = np.load('./samples/raw/' + str(j) + '_image.npy')
                        image = image[:, :, ::4]
                        mask = np.transpose(pred, (1, 2, 0))
                        folder_name = './metrics/' + letter[k] + '/' + str(nTrain) + '_TTA' + str(i) + '/' + str(
                            j) + '/'
                        mask_labels = np.load('./samples/raw/' + str(j) + '_labels.npy')
                        if j == 7:
                            mask_labels[:, :, 79] = mask_labels[:, :, 78]
                            mask_labels[:, :, 80] = mask_labels[:, :, 81]
                        mask_labels = mask_labels[:, :, ::4]
                        compute_bio_stats(image, mask, folder_name, mask_3d_labels=mask_labels)
                        # compute_bio_stats(image, mask, folder_name)
                    if not os.path.isfile(src_dir + '/predictions/' + str(j) + '_prediction_fusion_TTA_new.npy'):
                        np.save(src_dir + '/predictions/' + str(j) + '_prediction_fusion_TTA_new.npy',
                                np.zeros(pred.shape))
                    acc = np.load(src_dir + '/predictions/' + str(j) + '_prediction_fusion_TTA_new.npy')
                    acc = acc + pred
                    os.remove(src_dir + '/predictions/' + str(j) + '_prediction_TTA' + str(i) + '.npy')
                    acc = acc.astype(np.uint8)
                    np.save(src_dir + '/predictions/' + str(j) + '_prediction_fusion_TTA_new.npy', acc)
                    if i == 19:
                        np.save(src_dir + '/predictions/' + str(j) + '_prediction_fusion_TTA_new_20predictions.npy',
                                acc)
                        os.remove(src_dir + '/predictions/' + str(j) + '_prediction_fusion_TTA_new.npy')
    else:
        for i, value in enumerate(my_src_dir_list):
            my_src_dir = value
            tda_angle = 0
            intensity = 0
            noise_std = 15
            cv_predict_model(my_src_dir, en_TTA, data_dir='./samples/raw/', gpu=2, tda_angle=tda_angle,
                epoch_to_load=epoch_to_load, intensity=intensity, noise_std=noise_std, save_dir=save_dir[i])
            for j in [6, 7, 9, 10]:
                #src_dir = save_dir[i] + '/firstval600'
                src_dir = save_dir[i]
                pred = np.load(src_dir + '/predictions/' + str(j) + '_prediction_epoch' + str(epoch_to_load) + '_std_noise15.npy')
                if compute_metrics_en:
                    image = np.load('./samples/raw/' + str(j) + '_image.npy')
                    image = image[:, :, ::4]
                    mask = np.transpose(pred, (1, 2, 0))
                    folder_name = './metrics/' + letter[i] + '/'+str(i+1) + '_' + str(nTrain) + '_noise15/' + str(j) + '/'
                    mask_labels = np.load('./samples/raw/' + str(j) + '_labels.npy')
                    if j == 7:
                        mask_labels[:, :, 79] = mask_labels[:, :, 78]
                        mask_labels[:, :, 80] = mask_labels[:, :, 81]
                    mask_labels = mask_labels[:, :, ::4]
                    compute_bio_stats(image, mask, folder_name, mask_3d_labels=mask_labels)


def cv_train_model(params, params_entries, **kwargs):
    data_dir = kwargs.get('data_dir', None)
    gpu = kwargs.get('gpu', 0)
    images_file = kwargs.get('images_file', None)
    masks_file = kwargs.get('masks_file', None)
    previous_dir = kwargs.get('previous_dir', None)
    nTrainMax = kwargs.get('nTrainMax', 6*100)
    images_range = kwargs.get('images_range', None)

    # Save parameters and create results folder path
    results_name_short = params2name({k: params[k] for k in params_entries})
    dest_dir = './results/' + results_name_short
    save_params(params, dest_dir)

    # Run the cross validation training
    for i in [0]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        #cv = cv_index_generator(params, params['nTrain'], dest_dir, i, True)
        cv = cv_index_generator(params, nTrainMax, dest_dir, i, True)
        images = np.load(data_dir + images_file)
        masks = np.load(data_dir + masks_file)
        train_2d(params, cv, dest_dir, images=images, masks=masks, en_online=params['en_online'], gpu=gpu,
                        previous_dir=previous_dir, images_range=images_range)


def train_2d(params, cv, dest_dir, **kwargs):

    # Get kwargs
    previous_dir = kwargs.get('previous_dir', None)
    en_online = kwargs.get('en_online', 1)
    gpu = kwargs.get('gpu', 0)
    images = kwargs.get('images', None)
    masks = kwargs.get('masks', None)
    images_range = kwargs.get('images_range', None)

    # Set gpu, seed and time
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    time_start = time.clock()

    # Create folder to store the trained model
    crossvalidation_dir = dest_dir + '/firstval' + str(cv['val'][0])
    if previous_dir is not None:
        crossvalidation_previous_dir = previous_dir + '/firstval' + str(cv['val'][0])
    if not os.path.exists(crossvalidation_dir):
        os.mkdir(crossvalidation_dir)

    # Create or load model
    if previous_dir is None:
        print('Starting training from scratch.')
        params_previous = {'nb_epochs': 0}
        if params['model'] == 'unet_2d':
            model = unet_2d(params)
        elif params['model'] == 'unet_3d':
            model = unet_3d(params)
        elif params['model'] == 'munet_3d':
            model = munet_3d(params)
        hist_previous = None  # no history from previous model
    else:
        params_previous = pickle.load(open(previous_dir + '/params.p', "rb"))
        print('Resuming training of a model trained for ' + str(params_previous['nb_epochs']) + ' epochs.')
        hist_previous = pickle.load(open(crossvalidation_previous_dir + '/history.p', "rb"))  # history from previous model
        if params['model'] == 'unet_2d':
            co = {'dice_loss_2d': dice_loss_2d, 'dice_2d': dice_2d}
        elif params['model'] == 'unet_3d':
            co = {'dice_loss_3d': dice_loss_3d, 'dice_3d': dice_3d}
        elif params['model'] == 'munet_3d':
            co = {'dice_loss_3d': dice_loss_3d, 'dice_3d': dice_3d}
        model = models.load_model(crossvalidation_previous_dir + '/weights.h5', custom_objects=co)
        #plot_model(model, to_file='./model.png')

    # Train model
    if en_online:
        # Load data
        print(cv['train'])
        # list_train = cv['train']
        list_val = cv['val']
        train_images = images[images_range]
        train_masks = masks[images_range]
        val_images = images[list_val]
        val_masks = masks[list_val]
        val_images = val_images[::20]  # To accelerate the validation
        val_masks = val_masks[::20]

        # Normalize data
        norm_params = {}
        norm_file = '/export/home/jleger/Documents/segmentation/cartilage/results/model_unet_2d_modality_real5layers_' \
                    'partition_0_200_a_brightness_noise_wd_0_bn_1_nb_epochs_150_init_glorot_normal_n_layers_5' \
                    '/firstval600'
        norm_params = pickle.load(open(norm_file + '/norm_params.p', "rb"))
        #norm_params['mu'] = np.mean(train_images)
        #norm_params['sigma'] = np.std(train_images)

        pickle.dump(norm_params, open(crossvalidation_dir + '/norm_params.p', "wb"))
        train_images = (train_images - norm_params['mu']) / norm_params['sigma']
        val_images = (val_images - norm_params['mu']) / norm_params['sigma']

        print(norm_params['mu'])
        print(norm_params['sigma'])

        model_checkpoint = ModelCheckpoint(crossvalidation_dir + '/weights.{epoch:03d}.h5',
                                           verbose=1,
                                           monitor='val_' + params['loss'],
                                           save_best_only=False,
                                           save_weights_only=True,
                                           save_freq=9000)

        image_size = np.array([1024, 1024])
        training_generator = DataGeneratorTrainNew(params, train_images, train_masks, image_size, norm_params['sigma'],
                                                   norm_params['mu'])
        val_images = np.expand_dims(val_images, axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)
        history = model.fit_generator(generator=training_generator,
                                   validation_data=(val_images, val_masks),
                                   use_multiprocessing=False,
                                   workers=1,
                                   verbose=1,
                                   epochs=params['nb_epochs'] - params_previous['nb_epochs'],
                                   callbacks=[model_checkpoint])

    # Save model
    model.save(crossvalidation_dir + '/weights.h5')

    # Save training stats
    train_time = (time.clock() - time_start)
    np.save(crossvalidation_dir + '/train_time.npy', train_time)
    hist_new = history.history  # history for the new epochs
    if hist_previous is None:  # No previous model
        hist = hist_new
    elif hist_new == {}:
        hist = hist_previous
    else:
        hist = {}
        for key in hist_previous.keys():
            hist[key] = hist_previous[key] + hist_new[key]
    pickle.dump(hist, open(crossvalidation_dir + '/history.p', "wb"))
    save_history(hist, params, cv, dest_dir)


def cv_predict_model(src_dir, en_TTA, **kwargs):

    # Load params and create results folder path
    params = pickle.load(open(src_dir + '/params.p', 'rb'))
    nVal_old = params['nVal']
    params['nTrain'] = 6
    params['nVal'] = 6
    # Run the cross validation prediction
    for i in [0]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], None, i, True)
        crossvalidation_dir = src_dir + '/firstval600'
        #predict_model(crossvalidation_dir, cv['val'], params['loss'], **kwargs)
        predict_model(crossvalidation_dir, [6, 7, 9, 10], params['loss'], en_TTA, **kwargs)


def predict_model(src_dir, prediction_indices, loss_name, en_TTA, **kwargs):
    params = {'nTrain': 6 * 100,  # without data augmentation
                  'nVal': 6 * 100,
                  'nTest': 0,
                  'model': 'unet_2d',
                  'n_layers': 5,
                  'n_feat_maps': 16,
                  'batch_size': 5,
                  'nb_epochs': 150,
                  'lr': 1e-4,
                  'loss': 'dice_loss_2d',
                  'wd': 0,
                  'dropout': 0,
                  'bn': 1,
                  'en_online': 1,
                  'init': 'glorot_normal',
                  'modality': 'real5layers_partition_0_200_a_brightness5'}

    # Get kwargs
    en_online = kwargs.get('en_online', 1)
    gpu = kwargs.get('gpu', 2)
    images = kwargs.get('images', None)
    data_dir = kwargs.get('data_dir', None)
    batch_size = kwargs.get('batch_size', 1)
    en_save = kwargs.get('en_save', 1)
    prediction_type = kwargs.get('prediction_type', 'std')
    tda_angle = kwargs.get('tda_angle', 0)
    epoch_to_load = kwargs.get('epoch_to_load', 150)
    intensity = kwargs.get('intensity', 0)
    index = kwargs.get('index', 0)
    noise_std = kwargs.get('noise_std', 0)
    save_dir = kwargs.get('save_dir', src_dir)


    # Set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Get model and normalization parameters
    if loss_name == 'dice_loss_2d':
        co = {'dice_loss_2d': dice_loss_2d, 'dice_2d': dice_2d}
    elif loss_name == 'dice_loss_3d':
        co = {'dice_loss_3d': dice_loss_3d, 'dice_3d': dice_3d}
    elif loss_name == 'dice_loss_3d_multiresolution':
        co = {'dice_loss_3d': dice_loss_3d, 'dice_3d': dice_3d}
    #model = models.load_model(src_dir + '/weights.060.h5', custom_objects=co)
    model = unet_2d(params)
    #model.load_weights(src_dir + '/weights.150.h5')
    #model.load_weights(src_dir + '/weights.h5')
    if os.path.isfile(src_dir + '/weights.h5'):
        model.load_weights(src_dir + '/weights.h5')
    else:
        if epoch_to_load < 100:
            model.load_weights(src_dir + '/weights.0'+str(epoch_to_load)+'.h5')
        else:
            model.load_weights(src_dir + '/weights.'+str(epoch_to_load)+'.h5')

    norm_params = pickle.load(open(src_dir + '/norm_params.p', "rb"))

    # Make directories to store the predictions and metrics
    if not os.path.exists(src_dir + '/predictions'):
        os.makedirs(src_dir + '/predictions')

    # Predictions
    if en_online:
        for i, value in enumerate(prediction_indices):
            im_original = np.load(data_dir + '/' + str(value) + '_image.npy')
            im_original = im_original[:, :, ::4]
            im_original = im_original.astype(np.double)
            im_original = im_original + intensity
            im_original = im_original + np.random.normal(0, noise_std, im_original.shape)
            sh = im_original.shape
            im = np.zeros((sh[2], sh[0], sh[1]))
            for ind in range(sh[2]):
                im_slice = im_original[:, :, ind]
                #im_slice = im_slice+10
                #im_slice = ndimage.shift(im_slice, (1, 0))
                #im_slice = np.flip(im_slice, axis=1)
                im_slice = rotate(im_slice, tda_angle, reshape=False)
                #im_slice = np.transpose(im_slice)
                im[ind, :, :] = im_slice
            im = (im - norm_params['mu']) / norm_params['sigma']
            print(norm_params['sigma'])
            im = np.expand_dims(im, axis=-1)
            prediction = model.predict(im, batch_size, verbose=0)
            prediction_original = np.squeeze(prediction)
            prediction_resolution = np.zeros(prediction_original.shape)
            for ind in range(sh[2]):
                im_slice = prediction_original[ind, :, :]
                #im_slice = ndimage.shift(im_slice, (-1, 0))
                #im_slice = np.flip(im_slice, axis=1)
                im_slice = rotate(im_slice, -tda_angle, reshape=False)
                prediction_resolution[ind, :, :] = im_slice
            prediction_thr = np.zeros(prediction_resolution.shape)
            prediction_thr[prediction_resolution > 0.5] = 1
            prediction_thr = prediction_thr.astype(np.uint8)

            if en_save:
                if not os.path.exists(save_dir + '/predictions/'):
                    os.makedirs(save_dir + '/predictions/')
                if en_TTA:
                    np.save(save_dir + '/predictions/' + str(value) + '_prediction_TTA' + str(index) + '.npy', prediction_thr)
                else:
                    np.save(save_dir + '/predictions/' + str(value) + '_prediction_epoch' + str(epoch_to_load) + '_std_noise15.npy', prediction_thr)

            del im


