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
from training_inference import *
from generate_data import *
from functions import *

#############################
# Generate data             #
#############################

path_src = './raw/jpeg/'
path_dest = './samples/raw/'
generate_cropped_datasets(path_src, path_dest)
generate_partition(path_src, path_dest)


#############################
# Training                  #
#############################

# Configuration C(1,10)
nb_models = 25
nb_epochs = 9000
nTrain = 10
gpu = 0
images_range = (np.arange(0, 100, 10))
for i in range(nb_models):
    name = 'real5layers_partition_0_200_' + str(i) + 'dataset0_brightness_noise'
    train_segmenter_2d(name, gpu, nb_epochs, nTrain, images_range)

# Configuration C(2,5)
nb_models = 25
nb_epochs = 9000
nTrain = 10
gpu = 0
images_range = np.concatenate((np.arange(10, 100, 19), np.arange(220, 300, 19)))
for i in range(nb_models):
    name = 'real5layers_partition_0_200_' + str(i) + 'd_brightness_noise'
    train_segmenter_2d(name, gpu, nb_epochs, nTrain, images_range)

# Configuration C(2, 100)
nb_models = 25
nb_epochs = 450
nTrain = 200
gpu = 0
images_range = np.concatenate((np.arange(0, 100), np.arange(200, 300)))
for i in range(nb_models):
    name = 'real5layers_partition_0_200_' + str(i) + 'd_brightness_noise'
    train_segmenter_2d(name, gpu, nb_epochs, nTrain, images_range)

# Configuration C(8, 5)
nb_models = 25
nb_epochs = 2240
nTrain = 40
gpu = 0
images_range = np.concatenate((np.arange(10, 100, 19), np.arange(120, 200, 19), np.arange(210, 300, 19), np.arange(320, 400, 19),
                               np.arange(410, 500, 19), np.arange(520, 600, 19), np.arange(810, 900, 19), np.arange(1120, 1200, 19)))
for i in range(nb_models):
    name = 'real5layers_partition_0_200_' + str(i) + '_brightness_noise'
    train_segmenter_2d(name, gpu, nb_epochs, nTrain, images_range)

# Configuration C(8, 100)
nb_models = 25
nb_epochs = 112
nTrain = 800
gpu = 0
images_range = np.concatenate((np.arange(0, 600), np.arange(800, 900), np.arange(1100, 1200)))
for i in range(nb_models):
    name = 'real5layers_partition_0_200_' + str(i) + '_brightness_noise'
    train_segmenter_2d(name, gpu, nb_epochs, nTrain, images_range)


#############################
# Inference                 #
#############################

# Configuration C(1,10)
option = 15
en_TTA = True
predict_segmenter_2d(option, en_TTA)

# Configuration C(2,5)
option = 16
en_TTA = True
predict_segmenter_2d(option, en_TTA)

# Configuration C(2, 100)
option = 19
en_TTA = True
predict_segmenter_2d(option, en_TTA)

# Configuration C(8, 5)
option = 17
en_TTA = True
predict_segmenter_2d(option, en_TTA)

# Configuration C(8, 100)
option = 18
en_TTA = True
predict_segmenter_2d(option, en_TTA)


###############################
# Reproduce paper results     #
###############################

# Fusion at the prediction level using majority voting
save_fusion_ensemble_full()
for i in range(5):
    save_fusion_ensemble(i)
majority_voting_TTA_bio_stats()

# Fusion at the characteristic level using averaging
average_TTA_bio_stats()
average_ensemble_bio_stats()

# Computation of the characteristics on the annotations
run_compute_bio_stats_GT()

# Compute calibration
call_compute_posterior_ece()
compute_calibration_metrics()

# Compute uncertainty
build_epistemic_curve_2()
build_aleatoric_curve()

# Tables
write_latex_code_ensemble_TTA_not_dsc()
write_latex_code_softfusion_table_fusion_ensemble_TTA_both_mean_noise15()
write_latex_code_softfusion_table_fusion_ensemble_TTA_both_mv_noise15()
write_calibration_table()

# Figures
build_scatter_plot2()
run_plot_bio_stat_choose_models_1_3()
correlation_DSC_entropy()
save_figure_intro()
save_figures_entropy_uncert_seg()
