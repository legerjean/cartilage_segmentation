from core.utils_core import *
from core.train import *
import numpy as np
import matplotlib.gridspec as gridspec
from keras import models


def cv_predict_model(src_dir, **kwargs):
    # Load params and create results folder path
    params = pickle.load(open(src_dir + '/params.p', 'rb'))
    nVal_old = params['nVal']
    params['nTrain'] = 6
    params['nVal'] = 6
    # Run the cross validation prediction
    for i in [0]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], None, i, True)
        #crossvalidation_dir = src_dir + '/firstval' + str(cv['val'][0])
        crossvalidation_dir = src_dir + '/firstval300'
        predict_model(crossvalidation_dir, cv['val'], params['loss'], **kwargs)


def predict_model(src_dir, prediction_indices, loss_name, **kwargs):

    # Get kwargs
    en_online = kwargs.get('en_online', 1)
    gpu = kwargs.get('gpu', 1)
    images = kwargs.get('images', None)
    data_dir = kwargs.get('data_dir', None)
    batch_size = kwargs.get('batch_size', 1)
    en_save = kwargs.get('en_save', 1)
    prediction_type = kwargs.get('prediction_type', 'std')

    # Set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Get model and normalization parameters
    if loss_name == 'dice_loss_2d':
        co = {'dice_loss_2d': dice_loss_2d, 'dice_2d': dice_2d}
    elif loss_name == 'dice_loss_3d':
        co = {'dice_loss_3d': dice_loss_3d, 'dice_3d': dice_3d}
    elif loss_name == 'dice_loss_3d_multiresolution':
        co = {'dice_loss_3d': dice_loss_3d, 'dice_3d': dice_3d}
    model = models.load_model(src_dir + '/weights.h5', custom_objects=co)
    norm_params = pickle.load(open(src_dir + '/norm_params.p', "rb"))

    # Make directories to store the predictions and metrics
    if not os.path.exists(src_dir + '/predictions'):
        os.makedirs(src_dir + '/predictions')

    # Predictions
    if en_online:
        for i, value in enumerate(prediction_indices):
            print('hello')
            im_original = np.load(data_dir + '/' + str(value) + '_image.npy')
            sh = im_original.shape
            im = np.zeros((sh[2], sh[0], sh[1]))
            for ind in range(sh[2]):
                im[ind, :, :] = im_original[:, :, ind]
            im = (im - norm_params['mu']) / norm_params['sigma']
            print(im.shape)
            im = np.expand_dims(im, axis=-1)
            #im = np.expand_dims(im, axis=0)
            prediction = model.predict(im, batch_size, verbose=0)
            # for resolution_i in range(len(prediction)):  # in order to generalize to multiple outputs of the model
            #     prediction_resolution = prediction[resolution_i]
            #     prediction_resolution = np.squeeze(prediction_resolution)
            #     prediction_thr = np.zeros(prediction_resolution.shape)
            #     prediction_thr[prediction_resolution > 0.5] = 1
            #     prediction_thr = prediction_thr.astype(np.uint8)
            #     if en_save:
            #         np.save(src_dir + '/predictions/' + str(value) + '_prediction_' + prediction_type + '_' + str(resolution_i) + '.npy', prediction_thr)
            prediction_resolution = np.squeeze(prediction)
            prediction_thr = np.zeros(prediction_resolution.shape)
            prediction_thr[prediction_resolution > 0.5] = 1
            prediction_thr = prediction_thr.astype(np.uint8)
            if en_save:
                np.save(src_dir + '/predictions/' + str(value) + '_prediction.npy', prediction_thr)
            del im
    else:
        predictions = np.zeros(images.shape)
        for i, value in enumerate(prediction_indices):
            im = (images[i] - norm_params['mean']) / norm_params['sigma']
            im = np.expand_dims(im, axis=-1)
            im = np.expand_dims(im, axis=0)
            prediction = model.predict(im, batch_size, verbose=0)
            prediction = np.squeeze(prediction)
            prediction_thr = np.zeros(prediction.shape)
            prediction_thr[prediction > 0.5] = 1
            prediction_thr = prediction_thr.astype(np.uint8)
            del im
            predictions[i] = prediction_thr
        if en_save:
            np.save(src_dir + '/predictions/prediction.npy', predictions)


def cv_display_predictions_3d(src_dir, **kwargs):
    # Load params and create results folder path
    params = pickle.load(open(src_dir + '/params.p', 'rb'))

    # Run the cross validation prediction
    for i in [0]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], None, i, True)
        crossvalidation_dir = src_dir + '/firstval' + str(cv['val'][0])
        display_predictions_munet_3d(crossvalidation_dir, cv['val'], **kwargs)


def display_predictions_munet_3d(src_dir, prediction_indices, **kwargs):

    # Get kwargs
    en_online = kwargs.get('en_online', 1)
    images = kwargs.get('images', None)
    data_dir = kwargs.get('data_dir', None)
    prediction_type = kwargs.get('prediction_type', 'std')

    # Plot images
    if en_online:
        for i, value in enumerate(prediction_indices):
            for resolution_i in range(6):
                im = np.load(data_dir + '/' + str(value) + '_image.npy')
                gt = np.load(data_dir + '/' + str(value) + '_labels.npy')
                exponent = 5 - resolution_i
                step = 2 ** exponent
                im_low_res = im[::step, ::step, ::step]
                gt_low_res = gt[::step, ::step, ::step]
                prediction = np.load(src_dir + '/predictions/' + str(value) + '_prediction_' + prediction_type + '_' + str(resolution_i) + '.npy')

                sh = im_low_res.shape
                for j in range(sh[-1]):
                    fig = plt.figure(figsize=(30, 10))
                    spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
                    spec2.update(wspace=0.3, hspace=0.04)
                    font_size = 10

                    # Image
                    ax00 = fig.add_subplot(spec2[0, 0])
                    ax00.axis('off')
                    ax00.set_title('Image', fontsize=font_size)
                    ax00.imshow(im_low_res[:, :, j], cmap='gray')

                    # Manual contours
                    ax01 = fig.add_subplot(spec2[0, 1])
                    output_gt = draw_color_single(im_low_res[:, :, j].astype(np.double), gt_low_res[:, :, j], color_channel=0)
                    ax01.axis('off')
                    ax01.set_title('Manual contours', fontsize=font_size)
                    ax01.imshow(output_gt)

                    # Predictions
                    ax02 = fig.add_subplot(spec2[0, 2])
                    output_prediction = draw_color_single(im_low_res[:, :, j].astype(np.double), prediction[:, :, j], color_channel=2)
                    ax02.axis('off')
                    ax02.set_title('Predicted contours', fontsize=font_size)
                    ax02.imshow(output_prediction)

                    # Delete and save
                    if not os.path.exists(src_dir + '/predictions/predictions_images_' + prediction_type + '_' + str(resolution_i) + '/' + str(value)):
                        os.makedirs(src_dir + '/predictions/predictions_images_' + prediction_type + '_' + str(resolution_i) + '/' + str(value))
                    plt.savefig(src_dir + '/predictions/predictions_images_' + prediction_type + '_' + str(resolution_i) + '/' + str(value) + '/' + str(j) + '_prediction.png')
                    plt.show()
                    plt.close()
    else:
        print('Display offline not supported yet')


def display_predictions_3d(src_dir, prediction_indices, **kwargs):

    # Get kwargs
    en_online = kwargs.get('en_online', 1)
    images = kwargs.get('images', None)
    data_dir = kwargs.get('data_dir', None)
    prediction_type = kwargs.get('prediction_type', 'std')

    # Plot images
    if en_online:
        for i, value in enumerate(prediction_indices):
            im = np.load(data_dir + '/' + str(value) + '_image.npy')
            gt = np.load(data_dir + '/' + str(value) + '_labels.npy')
            prediction = np.load(src_dir + '/predictions/' + str(value) + '_prediction_' + prediction_type + '.npy')

            sh = im.shape
            for j in range(sh[-1]):
                fig = plt.figure(figsize=(30, 10))
                spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
                spec2.update(wspace=0.3, hspace=0.04)
                font_size = 10

                # Image
                ax00 = fig.add_subplot(spec2[0, 0])
                ax00.axis('off')
                ax00.set_title('Image', fontsize=font_size)
                ax00.imshow(im[:, :, j], cmap='gray')

                # Manual contours
                ax01 = fig.add_subplot(spec2[0, 1])
                output_gt = draw_color_single(im[:, :, j].astype(np.double), gt[:, :, j], color_channel=0)
                ax01.axis('off')
                ax01.set_title('Manual contours', fontsize=font_size)
                ax01.imshow(output_gt)

                # Predictions
                ax02 = fig.add_subplot(spec2[0, 2])
                output_prediction = draw_color_single(im[:, :, j].astype(np.double), prediction[:, :, j], color_channel=2)
                ax02.axis('off')
                ax02.set_title('Predicted contours', fontsize=font_size)
                ax02.imshow(output_prediction)

                # Delete and save
                if not os.path.exists(src_dir + '/predictions/predictions_images_' + prediction_type + '/' + str(value)):
                    os.makedirs(src_dir + '/predictions/predictions_images_' + prediction_type + '/' + str(value))
                plt.savefig(src_dir + '/predictions/predictions_images_' + prediction_type + '/' + str(value) + '/' + str(j) + '_prediction.png')
                plt.show()
                plt.close()
    else:
        print('Display offline not supported yet')

