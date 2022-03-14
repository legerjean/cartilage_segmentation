
import numpy as np
from core.utils_core import *
from keras.callbacks import ModelCheckpoint
import os
import time
from core.online_data_augmentation_new import *
from keras.preprocessing.image import ImageDataGenerator

import pickle
from tensorflow import random


def cv_train_model(params, params_entries, **kwargs):
    data_dir = kwargs.get('data_dir', None)

    # Save parameters and create results folder path
    results_name_short = params2name({k: params[k] for k in params_entries})
    dest_dir = './results/' + results_name_short
    save_params(params, dest_dir)

    # Run the cross validation training
    for i in [0]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], dest_dir, i, True)
        #train_model(params, cv, dest_dir, **kwargs)
        #images = np.load(data_dir + 'image_partition_100_200.npy')
        #masks = np.load(data_dir + 'mask_partition_100_200.npy')
        images = np.load(data_dir + 'image_partition_50_100.npy')
        masks = np.load(data_dir + 'mask_partition_50_100.npy')
        #images = np.load(data_dir + 'image_partition_0_50.npy')
        #masks = np.load(data_dir + 'mask_partition_0_50.npy')
        train_2d(params, cv, dest_dir, images=images, masks=masks, en_online=params['en_online'])


# TRAIN_MODEL allows to train a model from scratch or from a pretrained model
# - params: dictionary with the following fields 'nTrain', 'nVal', 'nTest', 'model', 'n_layers', 'n_feat_maps',
#                                                'batch_size', 'nb_epoch', 'lr', 'loss', 'wd', 'dropout',
#                                                'bn', 'modality'
# - cv: dictionary with the following fields 'train', 'val', 'test' where each field is an array of indices
# - dest_dir: path of the folder where the results will be stored
# - previous_dir: path of the folder where the results will be stored

# - gpu: index of the used gpu
# - results_path: path of the folder where the results will be stored
# - en_test: if set to 1, enables test data

# - images: array with shape [nb_images x shape_image], e.g. [100, 256, 256, 80] for 100 images
# - masks: array with shape [nb_masks x shape_mask], e.g. [100, 256, 256, 80] for 100 masks


def train_model(params, cv, dest_dir, **kwargs):

    # Get kwargs
    previous_dir = kwargs.get('previous_dir', None)
    en_online = kwargs.get('en_online', 1)
    gpu = kwargs.get('gpu', 1)
    images = kwargs.get('images', None)
    masks = kwargs.get('masks', None)
    data_dir = kwargs.get('data_dir', None)
    image_size = kwargs.get('image_size', None)

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
        model = load_model(crossvalidation_previous_dir + '/weights.h5', custom_objects=co)
        plot_model(model, to_file='./model.png')

    # Set model checkpoint
    model_checkpoint = ModelCheckpoint(crossvalidation_dir + '/weights.h5',
                                       verbose =1,
                                       monitor = 'val_loss',
                                       save_best_only = False,
                                       save_weights_only = False,
                                       period = 2)

    # Train model
    if en_online:
        # Compute normalization parameters
        # s1 = 0
        # s2 = 0
        # n = len(cv['train']) * np.prod(image_size)
        # for index in cv['train']:
        #     image = np.load(data_dir + '/' + str(index) + '_image.npy').astype(np.uint16)
        #     s1 = s1 + np.sum(image.flatten())
        #     print(s1)
        #     s2 = s2 + np.sum(np.square(image.flatten()))
        #     print(s2)
        # normalization_params = {'mean': s1 / n, 'std': np.sqrt((n * s2 - s1 ** 2) / (n * (n - 1)))}
        # pickle.dump(normalization_params, open(crossvalidation_dir + '/normalization_params.p', "wb"))

        train_images = images[cv['train']]

        # Compute normalization parameters
        normalization_params = {'mu': np.mean(train_images), 'sigma': np.std(train_images)}
        pickle.dump(normalization_params, open(crossvalidation_dir + '/normalization_params.p', "wb"))

        # Fit model
        if 'loss_weights' in params:
            training_generator = DataGeneratorTrain_multiresolution(params, normalization_params, cv['train'], image_size, data_dir)
            validation_generator = DataGeneratorVal_multiresolution(params, normalization_params, cv['val'], image_size, data_dir)
        else:
            training_generator = DataGeneratorTrain2(params, normalization_params, cv['train'], image_size, data_dir)
            validation_generator = DataGeneratorVal2(params, normalization_params, cv['val'], image_size, data_dir)
            print(int(len(cv['train'])/params['batch_size']))
        history = model.fit_generator(generator = training_generator,
                                   validation_data = validation_generator,
                                   use_multiprocessing = False,
                                   workers = 1,
                                   steps_per_epoch = int(len(cv['train'])/params['batch_size']),
                                   validation_steps = len(cv['val']),
                                   verbose = 1,
                                   epochs = params['nb_epochs']-params_previous['nb_epochs'],
                                   callbacks = [model_checkpoint])
    else:
        # Build training and validation sets
        train_images = images[cv['train']]
        train_masks = masks[cv['train']]
        val_images = images[cv['val']]
        val_masks = masks[cv['val']]

        # Compute normalization parameters
        normalization_params = {'mu': np.mean(train_images), 'sigma': np.std(train_images)}
        pickle.dump(normalization_params, open(crossvalidation_dir + '/normalization_params.p', "wb"))
        train_images = (train_images - normalization_params['mu']) / normalization_params['sigma']
        val_images = (val_images - normalization_params['mu']) / normalization_params['sigma']

        # Add dimension
        train_images = np.expand_dims(train_images, axis=-1)
        train_masks = np.expand_dims(train_masks, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)
        print(train_masks.shape)
        print(val_masks.shape)

        train_masks_ds2 = train_masks[:, ::2, ::2, ::2, :]
        train_masks_ds4 = train_masks_ds2[:, ::2, ::2, ::2, :]
        train_masks_ds6 = train_masks_ds4[:, ::2, ::2, ::2, :]
        train_masks_ds8 = train_masks_ds6[:, ::2, ::2, ::2, :]
        train_masks_ds10 = train_masks_ds8[:, ::2, ::2, ::2, :]
        train_masks_all = [train_masks, train_masks_ds2, train_masks_ds4, train_masks_ds6, train_masks_ds8,
                           train_masks_ds10]
        train_masks_all = train_masks_all[::-1]
        val_masks_ds2 = val_masks[:, ::2, ::2, ::2, :]
        val_masks_ds4 = val_masks_ds2[:, ::2, ::2, ::2, :]
        val_masks_ds6 = val_masks_ds4[:, ::2, ::2, ::2, :]
        val_masks_ds8 = val_masks_ds6[:, ::2, ::2, ::2, :]
        val_masks_ds10 = val_masks_ds8[:, ::2, ::2, ::2, :]
        val_masks_all = [val_masks, val_masks_ds2, val_masks_ds4, val_masks_ds6, val_masks_ds8, val_masks_ds10]
        val_masks_all = val_masks_all[::-1]

        # Fit model
        history = model.fit(train_images,
                         train_masks_all,
                         batch_size = params['batch_size'],
                         nb_epoch = params['nb_epochs'],
                         verbose = 2,
                         shuffle = True,
                         validation_data = (val_images, val_masks_all),
                         callbacks = [model_checkpoint])

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


def train_2d(params, cv, dest_dir, **kwargs):

    # Get kwargs
    previous_dir = kwargs.get('previous_dir', None)
    en_online = kwargs.get('en_online', 1)
    gpu = kwargs.get('gpu', 1)
    images = kwargs.get('images', None)
    masks = kwargs.get('masks', None)
    data_dir = kwargs.get('data_dir', None)
    image_size = kwargs.get('image_size', None)

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
        model = load_model(crossvalidation_previous_dir + '/weights.h5', custom_objects=co)
        plot_model(model, to_file='./model.png')

    # Set model checkpoint
    model_checkpoint = ModelCheckpoint(crossvalidation_dir + '/weights.h5',
                                       verbose =1,
                                       monitor = 'val_loss',
                                       save_best_only = False,
                                       save_weights_only = False,
                                       period = 2)

    # Train model
    if en_online:
        # Load data
        list_train = cv['train']
        list_val = cv['val']
        train_images = images[list_train]
        train_masks = masks[list_train]
        val_images = images[list_val]
        val_masks = masks[list_val]

        # Normalize data
        norm_params = {}
        norm_params['mu'] = np.mean(train_images)
        norm_params['sigma'] = np.std(train_images)
        pickle.dump(norm_params, open(crossvalidation_dir + '/norm_params.p', "wb"))
        train_images = (train_images - norm_params['mu']) / norm_params['sigma']
        val_images = (val_images - norm_params['mu']) / norm_params['sigma']

        model_checkpoint = ModelCheckpoint(crossvalidation_dir + '/weights.h5',
                                           verbose=1,
                                           monitor='val_' + params['loss'],
                                           save_best_only=False,
                                           save_weights_only=True,
                                           save_freq='epoch')

        image_size = np.array([1024, 1024])
        training_generator = DataGeneratorTrainNew(params, train_images, train_masks, image_size)
        val_images = np.expand_dims(val_images, axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)
        #validation_generator = DataGeneratorValReg(params, val_images, val_masks)
        hist = model.fit_generator(generator=training_generator,
                         validation_data=(val_images, val_masks),
                         use_multiprocessing=False,
                         workers=1,
                         #steps_per_epoch=int(len(cv['train']) / params['batch_size']),
                         #validation_steps=int(len(cv['val']) / params['batch_size']),
                         verbose=1,
                         epochs=params['nb_epochs'] - params_previous['nb_epochs'],
                         callbacks=[model_checkpoint])
    if en_online == 2:
        train_images = images[cv['train']]
        train_masks = masks[cv['train']]
        val_images = images[cv['val']]
        val_masks = masks[cv['val']]

        print(np.amax(train_images))
        print(np.amax(train_masks))

        # Compute normalization parameters
        normalization_params = {'mu': np.mean(train_images), 'sigma': np.std(train_images)}
        train_images = (train_images - normalization_params['mu']) / normalization_params['sigma']
        val_images = (val_images - normalization_params['mu']) / normalization_params['sigma']
        pickle.dump(normalization_params, open(crossvalidation_dir + '/normalization_params.p', "wb"))

        # Add dimension
        train_images = np.expand_dims(train_images, axis=-1)
        train_masks = np.expand_dims(train_masks, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)

        print(train_images.shape)
        print(train_masks.shape)
        print('hello')

        # Build generator
        data_gen_args = dict(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=0,
            width_shift_range=0,
            height_shift_range=0,
            horizontal_flip=True,
            vertical_flip=True)
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen.fit(train_images, augment=True, seed=seed)
        mask_datagen.fit(train_masks, augment=True, seed=seed)
        image_generator = image_datagen.flow(train_images, batch_size=params['batch_size'], seed=seed)
        mask_generator = mask_datagen.flow(train_masks, batch_size=params['batch_size'], seed=seed)

        # Fit model
        history = model.fit(zip(image_generator, mask_generator),
                            validation_data=(val_images, val_masks),
                            use_multiprocessing=False,
                            workers=1,
                            steps_per_epoch=int(len(cv['train']) / params['batch_size']),
                            validation_steps=int(len(cv['val']) / params['batch_size']),
                            verbose=1,
                            epochs=params['nb_epochs'] - params_previous['nb_epochs'],
                            callbacks=[model_checkpoint])
    else:
        # Build training and validation sets
        train_images = images[cv['train']]
        train_masks = masks[cv['train']]
        val_images = images[cv['val']]
        val_masks = masks[cv['val']]

        # Compute normalization parameters
        normalization_params = {'mu': np.mean(train_images), 'sigma': np.std(train_images)}
        pickle.dump(normalization_params, open(crossvalidation_dir + '/normalization_params.p', "wb"))
        train_images = (train_images - normalization_params['mu']) / normalization_params['sigma']
        val_images = (val_images - normalization_params['mu']) / normalization_params['sigma']

        # Add dimension
        train_images = np.expand_dims(train_images, axis=-1)
        train_masks = np.expand_dims(train_masks, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)
        print(train_masks.shape)
        print(val_masks.shape)

        # Fit model
        history = model.fit(train_images,
                            train_masks,
                            batch_size=params['batch_size'],
                            epochs=params['nb_epochs'],
                            verbose=2,
                            shuffle=True,
                            validation_data=(val_images, val_masks),
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