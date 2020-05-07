import os
import sys
import time
import numpy as np
from PIL import Image
import cv2

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
        "C:\\projects\\deep-human-action-recognition\\output\\ntu_baseline\\0429\\base_ntu_model_weights.hdf5",
        by_name=True)



cam = cv2.VideoCapture(0)

"""Start capturing and prediction"""

img_array = []
while True: 
        
    ret, frame = cam.read()
                    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
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
    pred_x_y_1080 = np.interp(pred_x_y_1, (0, 1), (0, 480))
          
    index = 0
    for x in pred_x_y_1080:
        img_out = cv2.circle(img, (int(x[0]), int(x[1])), 6, (255, 0, 0), -1)
        index = index + 1
        height, width, layers = img_out.shape
        size = (width,height)
        
        frame = None
    
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
    
    #Save and Show Frame
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    img_array.append(img_out)
    cv2.imshow('Recording KINECT Video Stream', img_out)   
        
    key = cv2.waitKey(1)
    if key == 27: 
        cv2.destroyAllWindows()
        
        out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        cam.release()
        break