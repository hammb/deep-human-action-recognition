import os
import sys
import time
import numpy as np
from PIL import Image
import cv2
import concurrent.futures
from collections import Counter

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

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

from collections import Counter
import winsound
logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
cfg = ModelConfig((num_frames,) + ntu_dataconf.input_shape, pa17j3d,
        # num_actions=[60], num_pyramids=8, action_pyramids=[5, 6, 7, 8],
        num_actions=[4], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=192, num_visual_features=192)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

start_lr = 0.01
action_weight = 0.1
#batch_size_mpii = 3
#batch_size_h36m = 4
#batch_size_ntu = 6 #1
batch_clips = 1 # 8/4


"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(
         "E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0530\\weights_3dp+benset_ar_035.hdf5",
         by_name=True)



"""Start capturing and prediction"""
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale              = 2
fontColor              = (0,0,0)
lineType               = 2

action_labels = ['Spritze', 'Anziehen', 'Ausziehen', 'Nichts']

"""Load kinect"""
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

img_array_all = []
preprocessed_frames = []
frames = []

winsound.Beep(1000, 5000)
start_time = time.time()

while True: 
    
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        
        preprocess_frame(kinect.get_last_color_frame())
          
    if start_time + 20 < time.time():
        kinect.close()
        break
    
num_anziehen = 0
an = False
while True: 
    
    # --- Getting frames and drawing
    
    if len(preprocessed_frames) >= 32:
        actions_in_one_sec = []
        frames_one_sec = preprocessed_frames[:32]
        del preprocessed_frames[:32]
        for _ in range(4):
            
            prediction = full_model.predict(np.expand_dims(frames_one_sec[:8], 0))
            del frames_one_sec[:8]
        
            actions_in_one_sec.append(np.argmax(prediction[11]))
            #actions_in_one_sec = action_labels[np.argmax(prediction[11])]
            
        most_common_action = np.argmax([Counter(actions_in_one_sec)[0],Counter(actions_in_one_sec)[1],Counter(actions_in_one_sec)[2],Counter(actions_in_one_sec)[3]])
        
        if most_common_action == 1:
            num_anziehen = num_anziehen + 1
        else:
            num_anziehen = 0
            
        if num_anziehen >= 3:
            an = True
        
        if most_common_action == 2:
            an = False
        
        for img in frames[:32]:
            
            img = np.uint8(img)
            cv2.putText(img, str(action_labels[most_common_action]), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
            
            if an:
                cv2.putText(img, "Schutz an", 
                    (600,100), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
                
            else:
                cv2.putText(img, "Schutz aus", 
                    (600,100), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
                
            img = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)
            img_array_all.append(img)
            cv2.imshow('Recording KINECT Video Stream', img)
            cv2.waitKey(1)
        del frames[:32]
        
    else:
        
        cv2.destroyAllWindows()
        break
            
out = cv2.VideoWriter('test6.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (400,400))
        
for i in range(len(img_array_all)):
    out.write(img_array_all[i])
out.release()
    
 
def preprocess_frame(frame):
    
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
    
    #resize to 256x256
    img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
    
    #normalize
    preprocessed_frames.append(cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F))
    
    frames.append(img)

    
        