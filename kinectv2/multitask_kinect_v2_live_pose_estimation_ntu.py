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

cfg = ModelConfig(mpii_dataconf.input_shape, pa17j3d, num_pyramids=8,
        action_pyramids=[], num_levels=4)

"""Build the full model"""
model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
model.load_weights(
        # 'output/ntu_spnet_trial-03-ft_replica_0ae2bf7/weights_3dp+ntu_ar_062.hdf5',
        "C:\\networks\\deephar\\output\\ntu_baseline\\0429\\base_ntu_model_weights.hdf5",
        by_name=True)



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
        
        prediction = model.predict(np.expand_dims(img_256_norm, 0))
        
        pred_x_y_z_1 = prediction[5][0]

        pred_x_y_1 = pred_x_y_z_1[:,0:2]
        pred_x_y_1080 = np.interp(pred_x_y_1, (0, 1), (0, 1080))
          
        index = 0
        for x in pred_x_y_1080:
            img_out = cv2.circle(img, (int(x[0]), int(x[1])), 6, (0, 0, 255), -1)
            index = index + 1
        
        cv2.imshow('Recording KINECT Video Stream', img_out)
        frame = None
        
        
    key = cv2.waitKey(1)
    if key == 27: 
        kinect.close()
        cv2.destroyAllWindows()
        break