import os
import sys
import time
import numpy as np
from PIL import Image

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

import deephar

from keras.utils.data_utils import get_file

from deephar.config import ntu_dataconf

sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader import *
from benset_dataloader import *

from deephar.models import reception
from deephar.models import action
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from ntu_tools import eval_singleclip_generator

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
#import annothelper

#annothelper.check_ntu_dataset()

weights_file = 'weights_AR_merge_NTU_v2.h5'
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/deephar/releases/download/v0.4/' \
        + weights_file
md5_hash = 'ff98d70a7f6bc5976cc11c7a5760e8b7'

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')


num_frames = 20
num_blocks = 4
batch_size = 2
depth_maps = 8
num_joints = 20
num_actions = 60
pose_dim = 3
input_shape = ntu_dataconf.input_shape

"""Build the pose estimation model."""
model_pe = reception.build(input_shape, num_joints, dim=pose_dim,
        num_blocks=num_blocks, depth_maps=depth_maps, ksize=(5, 5),
        concat_pose_confidence=False)

"""Build the full model using the previous pose estimation one."""
model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=pose_dim,
        num_context_per_joint=0, pose_net_version='v2')

"""Load pre-trained model."""
weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
        cache_subdir='models')
model.load_weights(weights_path)

"""Load Benset dataset."""
benset = Benset('datasets/Benset')
benset_seq = BatchLoader(benset.get_dataset_structure(), benset.get_train_data(),[0,0],1,num_frames)

x_batch, y_batch = benset_seq.__getitem__(0)
frame_list_groups = benset_seq.get_frame_list()

action_labels = ['drink water', 'eat meal/snack', 'brushing teeth',
        'brushing hair', 'drop', 'pickup', 'throw', 'sitting down',
        'standing up (from sitting position)', 'clapping', 'reading',
        'writing', 'tear up paper', 'wear jacket', 'take off jacket',
        'wear a shoe', 'take off a shoe', 'wear on glasses',
        'take off glasses', 'put on a hat/cap', 'take off a hat/cap',
        'cheer up', 'hand waving', 'kicking something',
        'put something inside pocket / take out something from pocket',
        'hopping (one foot jumping)', 'jump up',
        'make a phone call/answer phone', 'playing with phone/tablet',
        'typing on a keyboard', 'pointing to something with finger',
        'taking a selfie', 'check time (from watch)', 'rub two hands together',
        'nod head/bow', 'shake head', 'wipe face', 'salute',
        'put the palms together', 'cross hands in front (say stop)',
        'sneeze/cough', 'staggering', 'falling', 'touch head (headache)',
        'touch chest (stomachache/heart pain)', 'touch back (backache)',
        'touch neck (neckache)', 'nausea or vomiting condition',
        'use a fan (with hand or paper)/feeling warm',
        'punching/slapping other person', 'kicking other person',
        'pushing other person', 'pat on back of other person',
        'point finger at the other person', 'hugging other person',
        'giving something to other person', 'touch other person s pocket',
        'handshaking', 'walking towards each other',
        'walking apart from each other']

index = 0
for frame_lists in frame_list_groups: 
    for i in range(len(frame_lists)):
        first_frame = frame_lists[i].start
        last_frame = frame_lists[i].stop
        
        img = x_batch[index][:,first_frame,:,:,:][0]
        img = np.interp(img, (-1, 1), (0, 256))
        img = Image.fromarray(np.uint8(img))
        
        img.show()
        
        input = x_batch[index][:,first_frame:last_frame,:,:,:]
        prediction = model.predict(input)
        
    index = index + 1