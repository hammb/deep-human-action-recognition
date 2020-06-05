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

num_frames = 8
cfg = ModelConfig((num_frames,) + ntu_dataconf.input_shape, pa17j3d,
        # num_actions=[60], num_pyramids=8, action_pyramids=[5, 6, 7, 8],
        num_actions=[4], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=192, num_visual_features=192)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)



"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(
         "E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0601\\weights_3dp+benset_ar_020.hdf5",
         by_name=True)

"""Load kinect"""

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

"""Start capturing and prediction"""

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale              = 2
fontColor              = (0,0,0)
lineType               = 2

action_labels = ['Spritze', 'An', 'Aus',
        'Nichts']

frames = np.zeros((num_frames,) + ntu_dataconf.input_shape)
frame_counter = 0
prediction = [0]

end_pred = np.ones(60)
img_array_all = []
img_array_batch = []
init = 1
size = (1080,1080)

while True: 
    
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        time.sleep(0.2)
        #start_time = time.time() # start time of the loop
        
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
                  
        img_array_batch.append(img)
        
        if init:
            cv2.imshow('Recording KINECT Video Stream', img)
        
        #resize to 256x256
        img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
        #normalize
        img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
        
        frames[frame_counter, :, :, :] = img_256_norm
        
        frame_counter = frame_counter + 1
        frame = None
        
        
    
    if frame_counter == num_frames:
        init = 0
        pred = full_model.predict(np.expand_dims(frames, 0))
        
        prediction_action = pred[-1]
        prediction_pose = pred[5][0]
        
        img_of_batch_index = 0
        for img_of_batch in img_array_batch:
            
            #Print action on image
            cv2.putText(img_of_batch, str(action_labels[np.argmax(prediction_action[0])]), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
            
            #print joints on image
            pred_x_y_z_1 = prediction_pose[img_of_batch_index]

            pred_x_y_1 = pred_x_y_z_1[:,0:2]
            pred_x_y_1080 = np.interp(pred_x_y_1, (0, 1), (0, 1080))
          
            index = 0
            for x in pred_x_y_1080:
                img_of_batch = cv2.circle(img_of_batch, (int(x[0]), int(x[1])), 6, (0, 0, 255), -1)
                index = index + 1
            
            #Wirbelsäule
    
            x1 = int(pred_x_y_1080[16][0])
            y1 = int(pred_x_y_1080[16][1])
                
            x2 = int(pred_x_y_1080[1][0])
            y2 = int(pred_x_y_1080[1][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,255,0), 2)
        
            x1 = int(pred_x_y_1080[16][0])
            y1 = int(pred_x_y_1080[16][1])
                
            x2 = int(pred_x_y_1080[0][0])
            y2 = int(pred_x_y_1080[0][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,255,0), 2)
        
            x1 = int(pred_x_y_1080[1][0])
            y1 = int(pred_x_y_1080[1][1])
                
            x2 = int(pred_x_y_1080[2][0])
            y2 = int(pred_x_y_1080[2][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,255,0), 2)
        
            x1 = int(pred_x_y_1080[3][0])
            y1 = int(pred_x_y_1080[3][1])
                
            x2 = int(pred_x_y_1080[2][0])
            y2 = int(pred_x_y_1080[2][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,255,0), 2)
        
        
            #linker Arm
            x1 = int(pred_x_y_1080[4][0])
            y1 = int(pred_x_y_1080[4][1])
                
            x2 = int(pred_x_y_1080[1][0])
            y2 = int(pred_x_y_1080[1][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (255,0,0), 2)
        
            x1 = int(pred_x_y_1080[4][0])
            y1 = int(pred_x_y_1080[4][1])
                
            x2 = int(pred_x_y_1080[6][0])
            y2 = int(pred_x_y_1080[6][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (255,0,0), 2)
        
            x1 = int(pred_x_y_1080[8][0])
            y1 = int(pred_x_y_1080[8][1])
                
            x2 = int(pred_x_y_1080[6][0])
            y2 = int(pred_x_y_1080[6][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (255,0,0), 2)
            #Rechter Arm
            x1 = int(pred_x_y_1080[5][0])
            y1 = int(pred_x_y_1080[5][1])
                
            x2 = int(pred_x_y_1080[1][0])
            y2 = int(pred_x_y_1080[1][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,0,255), 2)
        
            x1 = int(pred_x_y_1080[5][0])
            y1 = int(pred_x_y_1080[5][1])
                
            x2 = int(pred_x_y_1080[7][0])
            y2 = int(pred_x_y_1080[7][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,0,255), 2)
        
            x1 = int(pred_x_y_1080[7][0])
            y1 = int(pred_x_y_1080[7][1])
                
            x2 = int(pred_x_y_1080[9][0])
            y2 = int(pred_x_y_1080[9][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,0,255), 2)
        
            #linkes bein
            x1 = int(pred_x_y_1080[0][0])
            y1 = int(pred_x_y_1080[0][1])
                
            x2 = int(pred_x_y_1080[10][0])
            y2 = int(pred_x_y_1080[10][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (255,255,0), 2)
        
            x1 = int(pred_x_y_1080[10][0])
            y1 = int(pred_x_y_1080[10][1])
                
            x2 = int(pred_x_y_1080[12][0])
            y2 = int(pred_x_y_1080[12][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (255,255,0), 2)
        
            x1 = int(pred_x_y_1080[12][0])
            y1 = int(pred_x_y_1080[12][1])
                
            x2 = int(pred_x_y_1080[14][0])
            y2 = int(pred_x_y_1080[14][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (255,255,0), 2)
            #Rechtes Bein
        
            x1 = int(pred_x_y_1080[0][0])
            y1 = int(pred_x_y_1080[0][1])
                
            x2 = int(pred_x_y_1080[11][0])
            y2 = int(pred_x_y_1080[11][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,255,255), 2)
        
            x1 = int(pred_x_y_1080[11][0])
            y1 = int(pred_x_y_1080[11][1])
                
            x2 = int(pred_x_y_1080[13][0])
            y2 = int(pred_x_y_1080[13][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,255,255), 2)
        
            x1 = int(pred_x_y_1080[13][0])
            y1 = int(pred_x_y_1080[13][1])
                
            x2 = int(pred_x_y_1080[15][0])
            y2 = int(pred_x_y_1080[15][1])
                
            img_of_batch = cv2.line(img_of_batch, (x1, y1), (x2, y2), (0,255,255), 2)
            
            
            cv2.imshow('Recording KINECT Video Stream', img_of_batch)
            img_of_batch_index = img_of_batch_index + 1
            img_array_all.append(img_of_batch)
        
        
        frames = np.zeros((num_frames,) + ntu_dataconf.input_shape)
        frame_counter = 0
        img_array_batch = []
        
        
    
    
    key = cv2.waitKey(1)
    if key == 27: 
        kinect.close()
        cv2.destroyAllWindows()
        
        out = cv2.VideoWriter('test3.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array_all)):
            out.write(img_array_all[i])
        out.release()
        
        break