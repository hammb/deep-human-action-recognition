import os
import sys

import numpy as np
import json

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

import deephar

from deephar.config import ModelConfig
from deephar.config import DataConfig
dconf = DataConfig(scales=[0.9], hflips=[0], chpower=[1])

from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.models import spnet
from deephar.utils import *

from keras.models import Model

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
#from datasetpath import datasetpath

from generic import get_bbox_from_poses

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

cfg = ModelConfig(dconf.input_shape, pa16j2d, num_pyramids=8, num_levels=4)

"""Load dataset"""
#dpath = datasetpath('Penn_Action')
penn = PennAction('datasets/PennAction', dconf, poselayout=pa16j2d, topology='frames',
        use_gt_bbox=False)

"""Build and compile the network."""
model = spnet.build(cfg)
model.load_weights(
        'output/weights_AR_merge_ep074_26-10-17.h5')

"""Squeeze the model for only one output."""
model = Model(model.input, model.outputs[-1])


def predict_frame_bboxes(mode):
    bboxes = {}

    num_samples = penn.get_length(mode)

    for i in range(num_samples):
        printnl('%d: %06d/%06d' % (mode, i+1, num_samples))

        data = penn.get_data(i, mode)
        poses = model.predict(np.expand_dims(data['frame'], axis=0))
        bbox = get_bbox_from_poses(poses, data['afmat'], scale=1.5)
        seq_idx = data['seq_idx']
        f = data['frame_list'][0]
        bboxes['%d.%d' % (seq_idx, f)] = bbox.astype(int).tolist()

    return bboxes

bbox_tr = predict_frame_bboxes(TRAIN_MODE)
bbox_te = predict_frame_bboxes(TEST_MODE)
bbox_val = predict_frame_bboxes(VALID_MODE)

jsondata = [bbox_te, bbox_tr, bbox_val]

filename = os.path.join('datasets/PennAction', 'penn_pred_bboxess.json')
with open(filename, 'w') as fid:
    json.dump(jsondata, fid)

