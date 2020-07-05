import os
import sys
import time
import numpy as np
from PIL import Image

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

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

"""Load kinect"""

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

"""Start capturing and prediction"""
input_data = []

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale              = 2
fontColor              = (0,0,0)
lineType               = 2

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

frames = np.zeros((num_frames,) + pennaction_dataconf.input_shape)
frame_counter = 0
prediction = [0]

end_pred = np.ones(60)

while True: 
    
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        
        
        frame = kinect.get_last_color_frame()
        
        frame_counter = frame_counter + 1
        
        colourframe = np.reshape(frame, (2073600, 4))
        colourframe = colourframe[:,0:3]
        
        #extract then combine the RBG data
        colourframeR = colourframe[:,0]
        colourframeR = np.reshape(colourframeR, (1080, 1920))
        colourframeG = colourframe[:,1]
        colourframeG = np.reshape(colourframeG, (1080, 1920))        
        colourframeB = colourframe[:,2]
        colourframeB = np.reshape(colourframeB, (1080, 1920))
        
        framefullcolour = cv2.merge([colourframeR, colourframeG, colourframeB])
        framefullcolour = cv2.cvtColor(framefullcolour, cv2.COLOR_BGR2RGB)
        
        img = framefullcolour
                    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        x = int((len(img[0]) - len(img)) / 2)
        w = int(len(img))
        y = 0
        h = int(len(img))
        
        img = img[y:y+h, x:x+w]
        
        cv2.putText(img, str(action_labels[np.argmax(prediction[-1])]), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        
        cv2.imshow('Recording KINECT Video Stream', img)
        #resize to 256x256
        img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
        #normalize
        img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
        
        frames[frame_counter-1, :, :, :] = img_256_norm
        
        frame = None
        
    if frame_counter == num_frames:
        
        prediction = model.predict(np.expand_dims(frames, axis=0))
        end_pred = np.ones(60)
        for blocks in prediction:
            end_pred *= blocks[0]
        
        
        frames = np.zeros((num_frames,) + pennaction_dataconf.input_shape)
        frame_counter = 0
        input_data = []
    
    
    key = cv2.waitKey(1)
    if key == 27: 
        kinect.close()
        cv2.destroyAllWindows()
        break