import os
import numpy as np
import keras
import random
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk
import tensorflow as tf
from imageio import imwrite
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.compat.v1.disable_eager_execution()


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def image_transform_3d(image, shears, angles, shifts, order, dims, **kwargs):
    # Get kwargs
    flip = kwargs.get('flip', False)
    orientation = kwargs.get('orientation', (0, 1, 2))

    shear_matrix = np.array([[1,            shears[0],  shears[1],  0],
                             [shears[2],    1,          shears[3],  0],
                             [shears[4],    shears[5],  1,          0],
                             [0,            0,          0,          1]])

    shift_matrix = np.array([[1, 0, 0, shifts[0]],
                             [0, 1, 0, shifts[1]],
                             [0, 0, 1, shifts[2]],
                             [0, 0, 0, 1]])

    offset = np.array([[1, 0, 0, dims[0]],
                       [0, 1, 0, dims[1]],
                       [0, 0, 1, dims[2]],
                       [0, 0, 0, 1]])

    offset_opp = np.array([[1, 0, 0, -dims[0]],
                           [0, 1, 0, -dims[1]],
                           [0, 0, 1, -dims[2]],
                           [0, 0, 0, 1]])

    angles = np.deg2rad(angles)

    rotx = np.array([[1, 0,                 0,                  0],
                     [0, np.cos(angles[0]), -np.sin(angles[0]), 0],
                     [0, np.sin(angles[0]), np.cos(angles[0]),  0],
                     [0, 0,                 0,                  1]])

    roty = np.array([[np.cos(angles[1]),    0, np.sin(angles[1]),   0],
                     [0,                    1, 0,                   0],
                     [-np.sin(angles[1]),   0, np.cos(angles[1]),   0],
                     [0,                    0, 0,                   1]])

    rotz = np.array([[np.cos(angles[2]),    -np.sin(angles[2]), 0, 0],
                     [np.sin(angles[2]),    np.cos(angles[2]),  0, 0],
                     [0,                    0,                  1, 0],
                     [0,                    0,                  0, 1]])

    rotation_matrix = offset_opp.dot(rotz).dot(roty).dot(rotx).dot(offset)
    affine_matrix = shift_matrix.dot(rotation_matrix).dot(shear_matrix)
    image_t = ndimage.interpolation.affine_transform(image, affine_matrix, order=order, mode='nearest')

    if flip:
        image_t = image_t[:, :, ::-1]
    if orientation is not np.array([0, 1, 2]):
        image_t = np.transpose(image_t, orientation)

    return image_t


def image_transform_2d(image, shears, angles, shifts, brightness, contrast, noise_std, zoom, blur, **kwargs):
    order = kwargs.get('order', 3)
    dims = kwargs.get('dims', (1024, 1024))
    order_zoom = kwargs.get('order_zoom', 3)

    shear_matrix = np.array([[1, shears[0], 0],
                             [shears[1], 1, 0],
                             [0, 0, 1]])

    shift_matrix = np.array([[1, 0, shifts[0]],
                             [0, 1, shifts[1]],
                             [0, 0, 1]])

    offset = np.array([[1, 0, -int(dims[0] / 2)],
                       [0, 1, -int(dims[1] / 2)],
                       [0, 0, 1]])

    offset_opp = np.array([[1, 0, int(dims[0] / 2)],
                           [0, 1, int(dims[1] / 2)],
                           [0, 0, 1]])

    angle = np.deg2rad(angles)

    rotz = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

    rotation_matrix = offset_opp.dot(rotz).dot(offset)
    affine_matrix = rotation_matrix
    #affine_matrix = shift_matrix.dot(rotation_matrix)
    #affine_matrix = shift_matrix.dot(rotation_matrix).dot(shear_matrix)

    transformed_image = ndimage.interpolation.affine_transform(image, affine_matrix, order=order, mode='nearest')
            #transformed_image = np.maximum(np.minimum(transformed_image + brightness, 255), 0)
    transformed_image = transformed_image+brightness
    transformed_image = transformed_image+np.random.normal(0, noise_std, transformed_image.shape)
            # transformed_image = clipped_zoom(transformed_image, zoom, order=order_zoom)
            # transformed_image = gaussian_filter(transformed_image, blur)
            #transformed_image = contrast*(transformed_image-np.mean(transformed_image)) + np.mean(transformed_image)
            #print(np.amax(transformed_image))
    #transformed_image = image
    return transformed_image


class DataGeneratorTrainNew(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, params, training_images, training_masks, image_size, std, mean):
        'Initialization'
        self.sh = training_images.shape
        self.batch_size = params['batch_size']
        self.list_IDs = np.arange(self.sh[0])
        self.n_channels = 1
        self.shuffle = True
        self.on_epoch_end()
        self.training_images = training_images
        self.training_masks = training_masks
        self.image_size = image_size
        self.std = std
        self.mean = mean

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        dim = self.sh[1:]
        X = np.empty((self.batch_size, *dim, self.n_channels))
        Y = np.empty((self.batch_size, *dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            im_slice = self.training_images[ID]
            mask_slice = self.training_masks[ID]

            if len(self.image_size) == 2:
                shears = np.array([0 * random.uniform(-1, 1) for _ in range(2)])
                angle = np.array([180 * random.uniform(-1, 1)])
                #shifts = np.array([0 * random.uniform(-1, 1) * self.image_size[j] for j in range(2)])
                shifts = np.array([0, 0])
                brightness = random.uniform(-40, 40)/self.std
                noise_std = 15/self.std
                contrast = random.uniform(0.8, 1)
                #zoom = random.uniform(0.8, 1.2)
                zoom = random.uniform(0.5, 2)
                blur = random.uniform(0, 1)
                #flip = random.randint(0, 2)
                #im = im_slice
                #im = image_transform_2d(im_slice, shears, angle, shifts, brightness, contrast, noise_std, 1, blur,
                #                        order=3, dims=self.image_size)
                im = image_transform_2d(im_slice, shears, angle, shifts, brightness, contrast, noise_std, zoom, blur,
                                        order=3, dims=self.image_size)
                #im = image_transform_2d(im_slice, shears, angle, shifts, 0, contrast, 0, zoom, blur, order=3,
                #                        dims=self.image_size)
                #mask = mask_slice
                angle_2 = np.array([4 * random.uniform(-1, 1)])
                zoom_2 = random.uniform(0.97, 1.03)
                #mask = image_transform_2d(mask_slice, shears, angle+angle_2, shifts, 0, 1, 0, zoom_2, 0, order=0,
                #                          dims=self.image_size, order_zoom=0)
                mask = image_transform_2d(mask_slice, shears, angle, shifts, 0, 1, 0, zoom, 0, order=0,
                                          dims=self.image_size, order_zoom=0)
                # mask_1 = image_transform_2d(mask_slice, shears, angle, shifts, 0, 1, 0, 1, 0,
                #                           order=0,
                #                           dims=self.image_size, order_zoom=0)

            elif len(self.image_size) == 3:
                shears = np.array([0.02 * random.uniform(-1, 1) for _ in range(6)])
                angles = np.array([5 * random.uniform(-1, 1) for _ in range(3)])
                # angles = np.array([5 * random.uniform(-1, 1), 5 * random.uniform(-1, 1), 180 * random.uniform(-1, 1)])
                shifts = np.array([0.05 * random.uniform(-1, 1) * self.image_size[j] for j in range(3)])
                flip = random.choice([True, False])
                im = image_transform_3d(im, shears, angles, shifts, order=3, dims=self.image_size, flip=flip)
                mask = image_transform_3d(mask, shears, angles, shifts, order=0, dims=self.image_size, flip=flip)
            else:
                print('Error in online data augmentation: input dimensions not supported')

            #im_denorm = (im*self.std) + self.mean
            # im_slice_denorm = (im_slice*self.std) + self.mean
            #maxi = max(np.amax(im_denorm), np.amax(im_slice_denorm))
            #mini = min(np.amin(im_denorm), np.amin(im_slice_denorm))
            #maxi = np.amax(im_denorm)
            #mini = np.amin(im_denorm)
            #im_denorm = (im_denorm-mini)*255/(maxi-mini)
            # im_slice_denorm = (im_slice_denorm-mini)*255/(maxi-mini)
            # imwrite('/export/home/jleger/Documents/segmentation/cartilage/figure/online_im_generation/image_original'
            #         + str(i) + '.png', im_slice_denorm.astype(np.uint8))
            #imwrite('/export/home/jleger/Documents/segmentation/cartilage/figure/online_im_generation/im'+str(i)+'.png',
            #        im_denorm.astype(np.uint8))
            #imwrite(
            #    '/export/home/jleger/Documents/segmentation/cartilage/figure/online_im_generation/mask' + str(i) + '.png',
            #    ((np.double(mask)+np.double(mask_1))*255/2).astype(np.uint8))
            # imwrite(
            #     '/export/home/jleger/Documents/segmentation/cartilage/figure/online_im_generation/mask0' + str(
            #         i) + '.png',
            #     (mask * 255).astype(np.uint8))
            # imwrite(
            #     '/export/home/jleger/Documents/segmentation/cartilage/figure/online_im_generation/mask1' + str(
            #         i) + '.png',
            #     (mask_1 * 255).astype(np.uint8))

            X[i,] = np.expand_dims(im, axis=-1)  # change the expands dims if more than one input channel
            Y[i,] = np.expand_dims(mask, axis=-1)  # change the expands dims if more than one output channel

        return X, Y
