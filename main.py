# -*- coding: utf-8 -*-
import numpy as np
import argparse

from keras.preprocessing import image
from keras import backend as K
from keras_nets.utils.imagenet_utils import decode_predictions, preprocess_input

from keras_nets.AlexNet import AlexNet as AlexNetModel
from keras_nets.AlexNet_Original import AlexNet_Original as AlexNetOrigModel
from keras_nets.CaffeNet import CaffeNet as CaffeNetModel
from keras_nets.GoogLeNet import GoogLeNet as GoogLeNetModel
from keras_nets.inception_v3 import InceptionV3 as InceptionV3Model
from keras_nets.music_tagger_crnn import MusicTaggerCRNN as MusicTaggerCRNNModel
from keras_nets.resnet50 import ResNet50 as ResNet50Model
from keras_nets.vgg16 import VGG16 as VGG16Model
from keras_nets.vgg19 import VGG19 as VGG19Model
from keras_nets.xception import Xception as XceptionModel

model_choice = dict(AlexNet=AlexNetModel,
                    AlexNetOrig=AlexNetOrigModel,
                    CaffeNet=CaffeNetModel,
                    GoogLeNet=GoogLeNetModel,
                    InceptionV3=InceptionV3Model,
                    MusicTaggerCRNN=MusicTaggerCRNNModel,
                    ResNet50=ResNet50Model,
                    VGG16=VGG16Model,
                    VGG19=VGG19Model,
                    Xception=XceptionModel)


model_val = 'VGG16'

if __name__ == '__main__':
    """
        Compiles the respective Model of choice

    """
    model = model_choice[model_val](include_top=True)

    img_path = '1.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, 1))
