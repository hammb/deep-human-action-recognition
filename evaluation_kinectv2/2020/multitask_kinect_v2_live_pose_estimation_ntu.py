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

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

import deephar

from deephar.config import mpii_dataconf
from deephar.config import human36m_dataconf
from deephar.config import ntu_dataconf
from deephar.config import ModelConfig
from deephar.config import DataConfig

from deephar.data import Human36M
from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.trainer import MultiModelTrainer
from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
#from datasetpath import datasetpath

from h36m_tools import eval_human36m_sc_error
from ntu_tools import eval_multiclip_dataset

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 1
cfg = ModelConfig((num_frames,) + ntu_dataconf.input_shape, pa17j3d,
        # num_actions=[60], num_pyramids=8, action_pyramids=[5, 6, 7, 8],
        num_actions=[60], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=192, num_visual_features=192)

"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(
         "E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0429\\weights_3dp+ntu_ar_048.hdf5",
         by_name=True)

"""Split model to simplify evaluation."""
models = split_model(full_model, cfg, interlaced=False,
        model_names=['3DPose', '3DAction'])
"""Load kinect"""

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

"""Start capturing and prediction"""


while True: 
    
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        
        frame = kinect.get_last_color_frame()
        
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
        img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
        
        prediction = models[0].predict(np.expand_dims(np.expand_dims(img_256_norm, 0), 0))
        
        pred_x_y_z_1 = prediction[5][0][0]

        pred_x_y_1 = pred_x_y_z_1[:,0:2]
        pred_x_y_1080 = np.interp(pred_x_y_1, (0, 1), (0, 1080))
          
        index = 0
        for x in pred_x_y_1080:
            img_out = cv2.circle(img, (int(x[0]), int(x[1])), 6, (0, 0, 255), -1)
            index = index + 1
        
        
        #Wirbelsäule
    
        x1 = int(pred_x_y_1080[16][0])
        y1 = int(pred_x_y_1080[16][1])
            
        x2 = int(pred_x_y_1080[1][0])
        y2 = int(pred_x_y_1080[1][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
    
        x1 = int(pred_x_y_1080[16][0])
        y1 = int(pred_x_y_1080[16][1])
            
        x2 = int(pred_x_y_1080[0][0])
        y2 = int(pred_x_y_1080[0][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
    
        x1 = int(pred_x_y_1080[1][0])
        y1 = int(pred_x_y_1080[1][1])
            
        x2 = int(pred_x_y_1080[2][0])
        y2 = int(pred_x_y_1080[2][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
    
        x1 = int(pred_x_y_1080[3][0])
        y1 = int(pred_x_y_1080[3][1])
            
        x2 = int(pred_x_y_1080[2][0])
        y2 = int(pred_x_y_1080[2][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
    
    
        #linker Arm
        x1 = int(pred_x_y_1080[4][0])
        y1 = int(pred_x_y_1080[4][1])
            
        x2 = int(pred_x_y_1080[1][0])
        y2 = int(pred_x_y_1080[1][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
    
        x1 = int(pred_x_y_1080[4][0])
        y1 = int(pred_x_y_1080[4][1])
            
        x2 = int(pred_x_y_1080[6][0])
        y2 = int(pred_x_y_1080[6][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
    
        x1 = int(pred_x_y_1080[8][0])
        y1 = int(pred_x_y_1080[8][1])
            
        x2 = int(pred_x_y_1080[6][0])
        y2 = int(pred_x_y_1080[6][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
        #Rechter Arm
        x1 = int(pred_x_y_1080[5][0])
        y1 = int(pred_x_y_1080[5][1])
            
        x2 = int(pred_x_y_1080[1][0])
        y2 = int(pred_x_y_1080[1][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
        x1 = int(pred_x_y_1080[5][0])
        y1 = int(pred_x_y_1080[5][1])
            
        x2 = int(pred_x_y_1080[7][0])
        y2 = int(pred_x_y_1080[7][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
        x1 = int(pred_x_y_1080[7][0])
        y1 = int(pred_x_y_1080[7][1])
            
        x2 = int(pred_x_y_1080[9][0])
        y2 = int(pred_x_y_1080[9][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
        #linkes bein
        x1 = int(pred_x_y_1080[0][0])
        y1 = int(pred_x_y_1080[0][1])
            
        x2 = int(pred_x_y_1080[10][0])
        y2 = int(pred_x_y_1080[10][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
    
        x1 = int(pred_x_y_1080[10][0])
        y1 = int(pred_x_y_1080[10][1])
            
        x2 = int(pred_x_y_1080[12][0])
        y2 = int(pred_x_y_1080[12][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
    
        x1 = int(pred_x_y_1080[12][0])
        y1 = int(pred_x_y_1080[12][1])
            
        x2 = int(pred_x_y_1080[14][0])
        y2 = int(pred_x_y_1080[14][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
        #Rechtes Bein
    
        x1 = int(pred_x_y_1080[0][0])
        y1 = int(pred_x_y_1080[0][1])
            
        x2 = int(pred_x_y_1080[11][0])
        y2 = int(pred_x_y_1080[11][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
    
        x1 = int(pred_x_y_1080[11][0])
        y1 = int(pred_x_y_1080[11][1])
            
        x2 = int(pred_x_y_1080[13][0])
        y2 = int(pred_x_y_1080[13][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
    
        x1 = int(pred_x_y_1080[13][0])
        y1 = int(pred_x_y_1080[13][1])
            
        x2 = int(pred_x_y_1080[15][0])
        y2 = int(pred_x_y_1080[15][1])
            
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
        
        cv2.imshow('Recording KINECT Video Stream', img_out)
        frame = None
        
        
    key = cv2.waitKey(1)
    if key == 27: 
        kinect.close()
        cv2.destroyAllWindows()
        break