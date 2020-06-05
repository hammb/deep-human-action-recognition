import os
import sys
import time
import numpy as np
from PIL import Image
import cv2

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import deephar

from keras.utils.data_utils import get_file

from deephar.config import ntu_dataconf

from deephar.models import reception
from deephar.models import action
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from ntu_tools import eval_singleclip_generator

sys.path.append(os.path.join(os.getcwd(), 'datasets'))

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
"""Load kinect"""

final_path = "E:\\Bachelorarbeit-SS20\\datasets\\Benset256\\frames\\S00462C00000A00003\\00048.jpg"
                
#load Image
#img = T(Image.open(final_path))
img = cv2.imread(final_path,1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#resize to 256x256
img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
#normalize
img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)

prediction = model_pe.predict(np.expand_dims(img_256_norm, 0))[6][0]

pred_x_y = prediction[:,0:2]

pred_x_y = np.interp(pred_x_y, (0, 1), (0, 256))
  
color = [(0, 0, 0), (252, 136, 3), (252, 215, 3), (165, 252, 3), (3, 252, 11), 
 (3, 252, 169), (3, 152, 252), (3, 40, 252), (107, 3, 252), (181, 3, 252), 
 (252, 3, 123), (255, 255, 255), (0, 0, 0), (255, 0, 0), (252, 136, 3), (252, 215, 3), (165, 252, 3)]

count = 0
index = 0
for x in pred_x_y:
    
    if index == 10 or index == 11 or index == 18 or index == 19:
        index = index + 1
        continue
    
    img_out = cv2.circle(img, (int(x[0]), int(x[1])), 6, color[count], -1)
    count = count + 1
    index = index + 1
img = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)    
cv2.imshow('ImageWindow', img)
cv2.waitKey(0)
cv2.destroyAllWindows()