# -*- coding: utf-8 -*-
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras_nets.utils.Custom_layers import LRN2D
from keras_nets.utils.imagenet_utils import decode_predictions, preprocess_input, maybe_download

PRETRAINED_MODELS_PATH = '/home/carey/workspace/pretrained_models/keras/alexnet/'
PRETRAINED_MODELS_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/' #url error

TH_WEIGHTS = 'alexnet_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS= 'alexnet_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_NO_TOP = 'alexnet_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_NO_TOP = 'alexnet_weights_tf_dim_ordering_tf_kernels_notop.h5'

# global constants
NB_CLASS = 1000         # number of classes
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
LRN2D_norm = True       # whether to use batch normalization

def conv2D_lrn2d(x, nb_filter, nb_row, nb_col,
                 border_mode='same', subsample=(1, 1),
                 activation='relu', LRN2D_norm=True,
                 weight_decay=WEIGHT_DECAY):
    '''

        Info:
            Function taken from the Inceptionv3.py script keras github


            Utility function to apply to a tensor a module Convolution + lrn2d
            with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      bias=False,
                      dim_ordering=K.image_dim_ordering())(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=K.image_dim_ordering())(x)

    if LRN2D_norm:

        x = LRN2D(alpha=ALPHA, beta=BETA)(x)
        x = ZeroPadding2D(padding=(1, 1), dim_ordering=K.image_dim_ordering())(x)

    return x


def AlexNet(include_top=True, weights='imagenet',
          input_tensor=None):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    # Channel 1 - Convolution Net Layer 1
    x = conv2D_lrn2d(
        img_input, 3, 11, 11, subsample=(
            1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), dim_ordering=K.image_dim_ordering())(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=K.image_dim_ordering())(x)

    # Channel 1 - Convolution Net Layer 2
    x = conv2D_lrn2d(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=K.image_dim_ordering())(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=K.image_dim_ordering())(x)

    # Channel 1 - Convolution Net Layer 3
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=K.image_dim_ordering())(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=K.image_dim_ordering())(x)

    # Channel 1 - Convolution Net Layer 4
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=K.image_dim_ordering())(x)

    # Channel 1 - Convolution Net Layer 5
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=K.image_dim_ordering())(x)

    # Channel 1 - Cov Net Layer 6
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=K.image_dim_ordering())(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=K.image_dim_ordering())(x)

    # Channel 1 - Cov Net Layer 7
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Channel 1 - Cov Net Layer 8
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Final Channel - Cov Net 9
    x = Dense(output_dim=NB_CLASS,
              activation='softmax')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        print('K.image_dim_ordering:', K.image_dim_ordering())
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = maybe_download(TH_WEIGHTS,
                         PRETRAINED_MODELS_PATH,
                         PRETRAINED_MODELS_URL)
            else:
                weights_path = maybe_download(TH_WEIGHTS_NO_TOP,
                         PRETRAINED_MODELS_PATH,
                         PRETRAINED_MODELS_URL)
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = maybe_download(TF_WEIGHTS,
                         PRETRAINED_MODELS_PATH,
                         PRETRAINED_MODELS_URL)
            else:
                weights_path = maybe_download(TF_WEIGHTS_NO_TOP,
                         PRETRAINED_MODELS_PATH,
                         PRETRAINED_MODELS_URL)
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model



if __name__ == '__main__':
    model = AlexNet(include_top=True, weights='imagenet')

    img_path = '1.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
