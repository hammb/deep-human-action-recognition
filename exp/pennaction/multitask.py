import os
import sys

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

import cv2
import numpy as np

import deephar

from deephar.config import mpii_dataconf
from deephar.config import pennaction_dataconf
from deephar.config import ModelConfig

from deephar.data import MpiiSinglePerson
from deephar.data import PennAction
from deephar.data import Squads
from deephar.data import BatchLoader

from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
#from datasetpath import datasetpath

from mpii_tools import eval_singleperson_pckh
from penn_tools import eval_singleclip_generator
from penn_tools import eval_multiclip_dataset

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 16
cfg = ModelConfig((num_frames,) + pennaction_dataconf.input_shape, pa16j2d,
        num_actions=[15], num_pyramids=6, action_pyramids=[5, 6],
        num_levels=4, pose_replica=True,
        num_pose_features=160, num_visual_features=160)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)


"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(
        'output/weights_AR_merge_ep074_26-10-17.h5',
        by_name=True)

"""This call splits the model into its parts: pose estimation and action
recognition, so we can evaluate each part separately on its respective datasets.
"""
models = split_model(full_model, cfg, interlaced=False,
        model_names=['2DPose', '2DAction'])




use_bbox = False
squads_dataconf = pennaction_dataconf


"""Load Squads dataset."""

squad_seq = Squads('datasets/Squads', squads_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=use_bbox,
        clip_size=num_frames)



sys.path.append(os.path.join(os.getcwd(), 'exp/pennaction'))
from squads_predict_bboxes import Bbox

predict_bbox = Bbox(models[0], squad_seq)
predict_bbox.create_json_file()







data = squad_seq.get_data(0, 0, squad_frame_list)

pred = models[1].predict(np.expand_dims(data['frame'], axis=0))
























num_blocks = 4
batch_size = 2
num_joints = 16
num_actions = 15






"""Load datasets"""
mpii = MpiiSinglePerson('datasets/MPII', dataconf=mpii_dataconf,
        poselayout=pa16j2d)







"""Load PennAction dataset."""
penn_seq = PennAction('datasets/PennAction', pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=use_bbox,
        pred_bboxes_file=pred_bboxes_file,
        clip_size=num_frames)



 

"""Trick to pre-load validation samples from MPII."""
mpii_val = BatchLoader(mpii, ['frame'], ['pose', 'afmat', 'headsize'],
        VALID_MODE, batch_size=mpii.get_length(VALID_MODE), shuffle=False)
printnl('Pre-loading MPII validation data...')
[x_val], [p_val, afmat_val, head_val] = mpii_val[0]
 

"""Define a loader for PennAction test samples. """
penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TEST_MODE,
        batch_size=1, shuffle=False)

"""Evaluate on 2D action recognition (PennAction).""" 
s = eval_singleclip_generator(models[1], penn_te)
print ('Best score on PennAction (single-clip): ' + str(s))

s = eval_multiclip_dataset(models[1], penn_seq,
        subsampling=pennaction_dataconf.fixed_subsampling)
print ('Best score on PennAction (multi-clip): ' + str(s))

"""Evaluate on 2D pose estimation (MPII)."""
s = eval_singleperson_pckh(models[0], x_val, p_val[:, :, 0:2], afmat_val, head_val)
print ('Best score on MPII: ' + str(s))


##############################################################################################

import os

import numpy as np
import json
import time

from tensorflow.python.keras.callbacks import Callback

from deephar.data import BatchLoader
from deephar.utils import *

model = models[1]
penn = penn_seq
subsampling=pennaction_dataconf.fixed_subsampling
bboxes_file=None
logdir=None
verbose=1
