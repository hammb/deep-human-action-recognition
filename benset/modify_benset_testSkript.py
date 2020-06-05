import os
import sys
import time
import numpy as np
from PIL import Image
import pickle
import shutil
import cv2
import random

sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader_ar import *
from benset_dataloader_ar import *


dataset_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256"
num_action_predictions = 6
benset = Benset(dataset_path, num_action_predictions)

num_frames = 6
batch_size = 1

benset_seq = BatchLoader(benset, benset.get_train_data_keys(),benset.get_train_annotations(),batch_size,num_frames)

x_batch,y_batch = benset_seq.__next__()

"""
predictions = {}

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses.p", 'rb') as fp:
    predictions.update({0:pickle.load(fp)})

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses_prob.p", 'rb') as fp:
    predictions.update({1:pickle.load(fp)})
    
with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\pose_predictons.p", 'wb') as fp:
    pickle.dump(predictions, fp, protocol=pickle.HIGHEST_PROTOCOL)
"""
#Find Smallest Class, and randomly make all classes equal big
"""    
countA0 = []
countA1 = []
countA2 = []
countA3 = []

for sequence in dataset_structure:
    if sequence.find("A00000") == 12:
        countA0.append(sequence)
    if sequence.find("A00001") == 12:
        countA1.append(sequence)
    if sequence.find("A00002") == 12:
        countA2.append(sequence)
    if sequence.find("A00003") == 12:
        countA3.append(sequence)
        
smallest_class = min([len(countA0), len(countA1), len(countA2), len(countA3)])

for i in range(0,len(countA0)-smallest_class):
    index = random.randrange(0, len(countA0), 1)
    dataset_structure.pop(countA0[index])
    countA0.pop(index)
    
for i in range(0,len(countA1)-smallest_class):
    index = random.randrange(0, len(countA1), 1)
    dataset_structure.pop(countA1[index])
    countA1.pop(index)
    
for i in range(0,len(countA2)-smallest_class):
    index = random.randrange(0, len(countA2), 1)
    dataset_structure.pop(countA2[index])
    countA2.pop(index)
    
for i in range(0,len(countA3)-smallest_class):
    index = random.randrange(0, len(countA3), 1)
    dataset_structure.pop(countA3[index])
    countA3.pop(index)
    
"""

import os
import sys

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

import deephar

from deephar.config import mpii_dataconf
from deephar.config import human36m_dataconf
from deephar.config import ntu_dataconf
from deephar.config import ntu_pe_dataconf
from deephar.config import ModelConfig
from deephar.config import DataConfig

from deephar.data import MpiiSinglePerson
from deephar.data import Human36M
from deephar.data import PennAction
from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.losses import pose_regression_loss
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import keras.backend as K

from deephar.callbacks import SaveModel

from deephar.trainer import MultiModelTrainer
from deephar.models import compile_split_models
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
#from datasetpath import datasetpath

from mpii_tools import MpiiEvalCallback
from h36m_tools import H36MEvalCallback
from ntu_tools import NtuEvalCallback

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 6
cfg = ModelConfig((num_frames,) + ntu_dataconf.input_shape, pa17j3d,
        # num_actions=[60], num_pyramids=8, action_pyramids=[5, 6, 7, 8],
        num_actions=[60], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=192, num_visual_features=192)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

start_lr = 0.01
action_weight = 0.1
batch_size_mpii = 3
#batch_size_h36m = 4
batch_size_ntu = 6 #1
batch_clips = 3 # 8/4