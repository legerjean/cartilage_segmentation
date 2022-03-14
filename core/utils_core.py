from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Conv2D, MaxPooling2D, \
    Conv2DTranspose, Dropout, Add, Lambda, multiply, SpatialDropout3D, SpatialDropout2D, LeakyReLU, BatchNormalization
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import pickle
import os
import matplotlib.pyplot as plt
import csv
#from keras.models import load_model
from skimage.color import gray2rgb, rgb2gray, rgb2hsv, hsv2rgb
#from scipy.misc import imfilter


###############################
# Losses
###############################

def dice_loss_2d(y_true, y_pred):
    smooth = 1
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.math.divide(2 * intersection + smooth, card_y_true + card_y_pred + smooth)
    return -tf.reduce_mean(dices)


def dice_loss_2d_and_BCE(y_true, y_pred):
    smooth = 1
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.math.divide(2 * intersection + smooth, card_y_true + card_y_pred + smooth)
    return -tf.reduce_mean(dices) + binary_crossentropy(y_true, y_pred)


def dice_loss_3d(y_true, y_pred):
    smooth = 1
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.div(2 * intersection, card_y_true + card_y_pred + smooth)
    return -tf.reduce_mean(dices)


###############################
# Metrics
###############################

def dice_2d(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = K.cast(K.greater(y_pred_f, 0.5), K.floatx())
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.math.divide(2 * intersection, card_y_true + card_y_pred)
    return tf.reduce_mean(dices)


def dice_3d(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = K.cast(K.greater(y_pred_f, 0.5), K.floatx())
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.div(2 * intersection, card_y_true + card_y_pred)
    return tf.reduce_mean(dices)


###############################
# Models
###############################

def unet_2d(params):
    nb_layers = params['n_layers']
    nb_features = params['n_feat_maps']

    # Input layer
    inputs = Input(batch_shape=(None, None, None, 1))

    # Encoding part
    skips = []
    x = inputs
    for i in range(nb_layers):
        # First conv, bn, ReLu, dropout block
        x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Skip connection and maxpooling
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        #if i < 5:
        nb_features = nb_features*2

    # Bottleneck
    x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x
    x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

    # Decoding part
    for i in reversed(range(nb_layers)):
        #if i < 5:
        nb_features = int(nb_features / 2)

        # Upsampling and concatenate
        x = concatenate([Conv2DTranspose(nb_features, (2, 2), strides=(2, 2), padding='same',
                                         kernel_initializer=params['init'], bias_initializer=params['init'],
                                         kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x),
                         skips[i]], axis=3)

        # First conv, bn, ReLu, dropout block
        x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid',
                     kernel_initializer=params['init'], bias_initializer=params['init'])(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    if params['loss'] == 'dice_loss_2d':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss_2d, metrics=[dice_2d])

    return model


def unet_3d(params):
    nb_layers = params['n_layers']
    nb_features = params['n_feat_maps']

    # Input layer
    inputs = Input(batch_shape=(None, None, None, None, 1))

    # Encoding part
    skips = []
    x = inputs
    for i in range(nb_layers):
        # First conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Skip connection and maxpooling
        skips.append(x)
        # x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = MaxPooling3D(pool_size=(2, 2, 1))(x)
        nb_features = nb_features*2

    # Bottleneck
    x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x
    x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

    # Decoding part
    for i in reversed(range(nb_layers)):
        nb_features = int(nb_features / 2)

        # Upsampling and concatenate
        # x = concatenate([Conv3DTranspose(nb_features, (2, 2, 2), strides=(2, 2, 2), padding='same',
        #                                  kernel_initializer=params['init'], bias_initializer=params['init'],
        #                                  kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x),
        #                  skips[i]], axis=4)
        x = concatenate([Conv3DTranspose(nb_features, (2, 2, 1), strides=(2, 2, 1), padding='same',
                                         kernel_initializer=params['init'], bias_initializer=params['init'],
                                         kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x),
                         skips[i]], axis=4)

        # First conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

    # Output layer
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid',
                     kernel_initializer=params['init'], bias_initializer=params['init'])(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    if params['loss'] == 'dice_loss_3d':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss_3d, metrics=[dice_3d])

    return model


def conv_block(x, params, nb_features):
    x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
               kernel_initializer=params['init'], bias_initializer=params['init'],
               kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x
    return x


def munet_3d(params):
    nb_layers = params['n_layers']
    nb_features = params['n_feat_maps']

    # Input layer
    inputs = Input(batch_shape=(None, None, None, None, 1))

    # Encoding part
    skips = []
    x = inputs
    for i in range(nb_layers):
        # First conv, bn, ReLu, dropout block
        inputs_i = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = conv_block(x, params, nb_features)
        x = conv_block(x, params, nb_features)
        skips.append(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = concatenate([inputs_i, x], axis=4)
        nb_features = nb_features*2

    # Bottleneck
    outputs = []
    x = conv_block(x, params, nb_features)
    x = conv_block(x, params, nb_features)
    output_i = conv_block(x, params, nb_features)
    output_i = conv_block(output_i, params, nb_features)
    output_i = Conv3D(1, (1, 1, 1), activation='sigmoid',
                      kernel_initializer=params['init'], bias_initializer=params['init'])(output_i)
    outputs.append(output_i)

    # Decoding part
    for i in reversed(range(nb_layers)):
        nb_features = int(nb_features / 2)

        # Upsampling and concatenate
        x = concatenate([Conv3DTranspose(nb_features, (2, 2, 2), strides=(2, 2, 2), padding='same',
                                         kernel_initializer=params['init'], bias_initializer=params['init'],
                                         kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x),
                         skips[i]], axis=4)
        x = conv_block(x, params, nb_features)
        x = conv_block(x, params, nb_features)
        output_i = conv_block(x, params, nb_features)
        output_i = conv_block(output_i, params, nb_features)
        output_i = Conv3D(1, (1, 1, 1), activation='sigmoid',
                          kernel_initializer=params['init'], bias_initializer=params['init'])(output_i)
        outputs.append(output_i)

    model = Model(inputs=[inputs], outputs=outputs)
    if params['loss'] == 'dice_loss_3d':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss_3d, metrics=[dice_3d])
    elif params['loss']=='dice_loss_3d_multiresolution':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss_3d, loss_weights=params['loss_weights'], metrics=[dice_3d])

    return model

def autoencoder_3d(params):
    nb_layers = params['n_layers']
    nb_features = params['n_feat_maps']

    # Input layer
    inputs = Input(batch_shape=(None, None, None, None, 1))

    # Encoding part
    x = inputs
    for i in range(nb_layers):
        # First conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Skip connection and maxpooling
        # x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = MaxPooling3D(pool_size=(2, 2, 1))(x)
        nb_features = nb_features*2

    # Bottleneck
    x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x
    x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    encoded = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

    x = encoded
    for i in reversed(range(nb_layers)):
        nb_features = int(nb_features / 2)

        # Upsampling and concatenate
        # x = concatenate([Conv3DTranspose(nb_features, (2, 2, 2), strides=(2, 2, 2), padding='same',
        #                                  kernel_initializer=params['init'], bias_initializer=params['init'],
        #                                  kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x),
        #                  skips[i]], axis=4)
        x = Conv3DTranspose(nb_features, (2, 2, 1), strides=(2, 2, 1), padding='same',
                                         kernel_initializer=params['init'], bias_initializer=params['init'],
                                         kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)

        # First conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x
    decoded = x

    model = Model(inputs=[inputs], outputs=[decoded])
    if params['loss'] == 'dice_loss_3d':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss_3d, metrics=[dice_3d])

    return model


def decoder_3d(params):
    nb_layers = params['n_layers']
    nb_features = params['n_feat_maps']

    # Input layer
    inputs = Input(batch_shape=(None, None, None, None, 1))

    # Encoding part
    x = inputs
    for i in range(nb_layers):
        # First conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Skip connection and maxpooling
        # x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = MaxPooling3D(pool_size=(2, 2, 1))(x)
        nb_features = nb_features*2

    # Bottleneck
    x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x
    x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

    # Decoding part
    for i in reversed(range(nb_layers)):
        nb_features = int(nb_features / 2)

        # Upsampling and concatenate
        # x = concatenate([Conv3DTranspose(nb_features, (2, 2, 2), strides=(2, 2, 2), padding='same',
        #                                  kernel_initializer=params['init'], bias_initializer=params['init'],
        #                                  kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x),
        #                  skips[i]], axis=4)
        x = Conv3DTranspose(nb_features, (2, 2, 1), strides=(2, 2, 1), padding='same',
                                         kernel_initializer=params['init'], bias_initializer=params['init'],
                                         kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)

        # First conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv3D(nb_features, (3, 3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout3D(params['dropout'])(x) if params['dropout'] != 0 else x

    # Output layer
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid',
                     kernel_initializer=params['init'], bias_initializer=params['init'])(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    if params['loss'] == 'dice_loss_3d':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss_3d, metrics=[dice_3d])

    return model
###############################
# Misc
###############################

def save_history(hist, params, cv, results_path):
    x = range(1, len(hist['loss']) + 1)
    plt.figure(figsize=(12, 12))
    if params['loss'] == 'dice_loss_2d':
        plt.plot(x, hist['dice_2d'], 'o-', label='train')
        plt.plot(x, hist['val_dice_2d'], 'o-', label='val')
    elif params['loss'] == 'dice_loss_3d':
        plt.plot(x, hist['dice_3d'], 'o-', label='train')
        plt.plot(x, hist['val_dice_3d'], 'o-', label='val')
    elif params['loss'] == 'dice_loss_3d_multiresolution':
        # plt.plot(x, hist['conv3d_15_dice_3d'], 'o-', label='1_train')
        # plt.plot(x, hist['val_conv3d_15_dice_3d'], 'o-', label='1_val')
        # plt.plot(x, hist['conv3d_20_dice_3d'], 'o-', label='2_train')
        # plt.plot(x, hist['val_conv3d_20_dice_3d'], 'o-', label='2_val')
        # plt.plot(x, hist['conv3d_25_dice_3d'], 'o-', label='3_train')
        # plt.plot(x, hist['val_conv3d_25_dice_3d'], 'o-', label='3_val')
        # plt.plot(x, hist['conv3d_30_dice_3d'], 'o-', label='4_train')
        # plt.plot(x, hist['val_conv3d_30_dice_3d'], 'o-', label='4_val')
        # plt.plot(x, hist['conv3d_35_dice_3d'], 'o-', label='5_train')
        # plt.plot(x, hist['val_conv3d_35_dice_3d'], 'o-', label='5_val')
        # plt.plot(x, hist['conv3d_40_dice_3d'], 'o-', label='6_train')
        # plt.plot(x, hist['val_conv3d_40_dice_3d'], 'o-', label='6_val')
        plt.plot(x, hist['conv3d_11_dice_3d'], 'o-', label='1_train')
        plt.plot(x, hist['val_conv3d_11_dice_3d'], 'o-', label='1_val')
        plt.plot(x, hist['conv3d_16_dice_3d'], 'o-', label='2_train')
        plt.plot(x, hist['val_conv3d_16_dice_3d'], 'o-', label='2_val')
        plt.plot(x, hist['conv3d_21_dice_3d'], 'o-', label='3_train')
        plt.plot(x, hist['val_conv3d_21_dice_3d'], 'o-', label='3_val')
        plt.plot(x, hist['conv3d_26_dice_3d'], 'o-', label='4_train')
        plt.plot(x, hist['val_conv3d_26_dice_3d'], 'o-', label='4_val')
    plt.legend(loc='upper left')
    plt.ylabel('Loss')
    plt.grid(True)

    results_name = params2name(params)
    plt.savefig(results_path + '/firstval' + str(cv['val'][0]) + '/learning_curves.png')
    plt.close()


def params2name(params):
    results_name = ''
    for key in params.keys():
        results_name = results_name + key + '_' + str(params[key]) + '_'
    results_name = results_name[:-1]
    return results_name


def save_params(params, path):
    if not os.path.exists(path):
        os.mkdir(path)
    pickle.dump(params, open(path + '/params.p', "wb"))
    with open(path + '/params.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])


def cv_index_generator(params, nb_train_max, results_path, fold_index, en_print=True):
    ind_trainvaltest = nb_train_max + int((params['nVal'] + params['nTest']))
    ind_trainval = nb_train_max + int(params['nVal'])
    ind_train = nb_train_max
    listOfIndices = np.roll(np.arange(ind_trainvaltest), -fold_index * params['nVal'])
    trainList = listOfIndices[0:params['nTrain']]
    valList = listOfIndices[ind_train:ind_trainval]
    testList = listOfIndices[ind_trainval:ind_trainvaltest]
    cv = {'train': trainList, 'val': valList, 'test': testList, 'cvNum': fold_index}

    if en_print:
        print(cv['train'])
        print(cv['val'])
        print(cv['test'])

    if results_path is not None:
        if not os.path.exists(results_path + '/firstval' + str(cv['val'][0])):
            os.makedirs(results_path + '/firstval' + str(cv['val'][0]))
        pickle.dump(cv, open(results_path + '/firstval' + str(cv['val'][0]) + '/cv.p', "wb"))

    return cv


def concatenate_predictions(predict_set, results_name_short, shape_images):
    # Load params
    results_path = './results/' + results_name_short
    with open(results_path + '/params.p', 'rb') as handle:
        params = pickle.load(handle)

    # Run the cross validation prediction
    predictions = np.zeros(shape_images)
    for i in [0, 1, 2]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], results_path, i, True)
        if predict_set == 'val':
            predictions_fold = np.load(results_path + '/firstval' + str(cv['val'][0]) + '/predictions.npy')
            predictions[cv['val']] = predictions_fold
        else:
            print('ERROR: concatenation not on the validation set is not yet supported')
    predictions = predictions.astype(np.uint8)
    np.save(results_path + '/predictions.npy', predictions)


def draw_color_single(image_in, labels_in, color_channel=0, **kwargs):  # image in [0, 255.0] input in double and color as well

    alpha = kwargs.get('alpha', 0.2)
    img = image_in
    rows, cols = img.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[:, :, color_channel] = labels_in

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = rgb2hsv(img_color)
    color_mask_hsv = rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = hsv2rgb(img_hsv)
    img_masked = img_masked.astype(int)

    return img_masked


def draw_color_single_line(image_in, labels_in, entropy_mask, color_channel=0, **kwargs):  # image in [0, 255.0] input in double and color as well

    alpha = kwargs.get('alpha', 0.2)
    img = image_in
    rows, cols = img.shape

    # Construct RGB version of grey-level image
    img_r = np.copy(img)
    #img_r[entropy_mask==1] = 0
    img_r[labels_in==1] = 255
    img_g = np.copy(img)
    #img_g[entropy_mask==1] = 0
    img_g[labels_in==1] = 0
    img_b = np.copy(img)
    #img_b[entropy_mask==1] = 255
    img_b[labels_in==1] = 0
    img_color = np.dstack((img_r, img_g, img_b))
    img_color = img_color.astype(int)

    return img_color


def draw_color_two(image_in, labels_in, labels_in2, color_channel=0, color_channel2=1, **kwargs):  # image in [0, 255.0] input in double and color as well

    alpha = kwargs.get('alpha', 0.2)
    img = image_in
    rows, cols = img.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[:, :, color_channel] = labels_in
    color_mask[:, :, color_channel2] = labels_in2

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = rgb2hsv(img_color)
    color_mask_hsv = rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = hsv2rgb(img_hsv)
    img_masked = img_masked.astype(int)

    return img_masked


def draw_color_three(image_in, labels_in, labels_in2, labels_in3, color_channel=0, color_channel2=1, color_channel3=2, **kwargs):  # image in [0, 255.0] input in double and color as well

    alpha = kwargs.get('alpha', 0.2)
    img = image_in
    rows, cols = img.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[:, :, color_channel] = labels_in
    color_mask[:, :, color_channel2] = labels_in2
    color_mask[:, :, color_channel3] = labels_in3

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = rgb2hsv(img_color)
    color_mask_hsv = rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = hsv2rgb(img_hsv)
    img_masked = img_masked.astype(int)

    return img_masked


def mask2contours(mask):
    sh = mask.shape
    contours = np.zeros((sh[0], sh[1], sh[2]))
    n_slices = sh[2]

    for s in range(n_slices):
        if s > 0 and np.sum(mask[:, :, s - 1].flatten()) == 0:
            contours[:, :, s] = mask[:, :, s]

        elif s < (n_slices - 1) and np.sum(mask[:, :, s + 1].flatten()) == 0:
            contours[:, :, s] = mask[:, :, s]
        else:
            diff = np.abs(mask[:, :, s] * 1 - mask[:, :, s - 1] * 1) > 0
            imf = imfilter(mask[:, :, s].astype('int'), 'find_edges')
            if np.sum(diff.flatten() > 0):
                contours[:, :, s] = (diff + imf) > 0
            else:
                contours[:, :, s] = imf
    return contours
