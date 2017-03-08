# -*- coding: utf-8 -*-
import os
import numpy as np
import json

from keras.utils.data_utils import get_file
from keras import backend as K

CLASS_INDEX = None
IMAGENET_CLASS_INDEX_PATH = '../image_models/'
IMAGENET_CLASS_INDEX_URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/'
IMAGENET_CLASS_INDEX = 'imagenet_class_index.json'

def maybe_download(filename, work_directory, SOURCE_URL):
	"""Download the data from website, unless it's already here."""
	if not os.path.exists(work_directory):
		os.mkdir(work_directory)
	filepath = os.path.join(work_directory, filename)
	if not os.path.exists(filepath):
		print('file \'' + filepath + '\' is not existed, downloading...')
		filepath = get_file(filename,
                                        SOURCE_URL + filename,
                                        cache_subdir=work_directory)
		statinfo = os.stat(filepath)
		print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
	return filepath

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = maybe_download(IMAGENET_CLASS_INDEX,
                         IMAGENET_CLASS_INDEX_PATH,
                         IMAGENET_CLASS_INDEX_URL)
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results
