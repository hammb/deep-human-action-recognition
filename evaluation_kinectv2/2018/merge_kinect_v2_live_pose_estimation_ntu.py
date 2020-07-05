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
size = (1080,1080)
img_array = []
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
        
        prediction = model_pe.predict(np.expand_dims(img_256_norm, 0))
        
        pred_x_y_z_1 = prediction[6][0]

        pred_x_y_1 = pred_x_y_z_1[:,0:2]
        pred_x_y_1080 = np.interp(pred_x_y_1, (0, 1), (0, 1080))
        
        joints = []
        for x in prediction[7][0]:
            if x[0] > 0.5:
                joints.append(1)
            else:
                joints.append(0)
             
        index = 0
        for x in pred_x_y_1080:
            if prediction[7][0][index][0] > 0.5:  
                img_out = cv2.circle(img, (int(x[0]), int(x[1])), 6, (0, 0, 255), -1)
                
            index = index + 1
        
        
        #Wirbelsäule
        if joints[3] and joints[2]:
            x1 = int(pred_x_y_1080[3][0])
            y1 = int(pred_x_y_1080[3][1])
            
            x2 = int(pred_x_y_1080[2][0])
            y2 = int(pred_x_y_1080[2][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
        
        if joints[2] and joints[1]:
            x1 = int(pred_x_y_1080[2][0])
            y1 = int(pred_x_y_1080[2][1])
            
            x2 = int(pred_x_y_1080[1][0])
            y2 = int(pred_x_y_1080[1][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
            
        if joints[1] and joints[0]:
            x1 = int(pred_x_y_1080[1][0])
            y1 = int(pred_x_y_1080[1][1])
            
            x2 = int(pred_x_y_1080[0][0])
            y2 = int(pred_x_y_1080[0][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
            
        #linker Arm
        
        if joints[4] and joints[1]:
            x1 = int(pred_x_y_1080[1][0])
            y1 = int(pred_x_y_1080[1][1])
            
            x2 = int(pred_x_y_1080[4][0])
            y2 = int(pred_x_y_1080[4][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
            
        if joints[4] and joints[6]:
            x1 = int(pred_x_y_1080[6][0])
            y1 = int(pred_x_y_1080[6][1])
            
            x2 = int(pred_x_y_1080[4][0])
            y2 = int(pred_x_y_1080[4][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
        
        if joints[6] and joints[8]:
            x1 = int(pred_x_y_1080[6][0])
            y1 = int(pred_x_y_1080[6][1])
            
            x2 = int(pred_x_y_1080[8][0])
            y2 = int(pred_x_y_1080[8][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
        
        if joints[10] and joints[8]:
            x1 = int(pred_x_y_1080[8][0])
            y1 = int(pred_x_y_1080[8][1])
            
            x2 = int(pred_x_y_1080[10][0])
            y2 = int(pred_x_y_1080[10][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,0), 2)
        
        #Rechter Arm
        
        if joints[1] and joints[5]:
            x1 = int(pred_x_y_1080[1][0])
            y1 = int(pred_x_y_1080[1][1])
            
            x2 = int(pred_x_y_1080[5][0])
            y2 = int(pred_x_y_1080[5][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
        
        if joints[5] and joints[7]:
            x1 = int(pred_x_y_1080[7][0])
            y1 = int(pred_x_y_1080[7][1])
            
            x2 = int(pred_x_y_1080[5][0])
            y2 = int(pred_x_y_1080[5][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
        
        if joints[7] and joints[9]:
            x1 = int(pred_x_y_1080[7][0])
            y1 = int(pred_x_y_1080[7][1])
            
            x2 = int(pred_x_y_1080[9][0])
            y2 = int(pred_x_y_1080[9][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
            
            
        if joints[11] and joints[9]:
            x1 = int(pred_x_y_1080[1][0])
            y1 = int(pred_x_y_1080[11][1])
            
            x2 = int(pred_x_y_1080[9][0])
            y2 = int(pred_x_y_1080[9][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
        
        #Hüfte
        
        if joints[0] and joints[13]:
            x1 = int(pred_x_y_1080[0][0])
            y1 = int(pred_x_y_1080[0][1])
            
            x2 = int(pred_x_y_1080[13][0])
            y2 = int(pred_x_y_1080[13][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 2)
            
        if joints[0] and joints[12]:
            x1 = int(pred_x_y_1080[0][0])
            y1 = int(pred_x_y_1080[0][1])
            
            x2 = int(pred_x_y_1080[12][0])
            y2 = int(pred_x_y_1080[12][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 2)
            
            
        #Linkes Bein
        if joints[14] and joints[12]:
            x1 = int(pred_x_y_1080[14][0])
            y1 = int(pred_x_y_1080[14][1])
            
            x2 = int(pred_x_y_1080[12][0])
            y2 = int(pred_x_y_1080[12][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
            
        if joints[16] and joints[14]:
            x1 = int(pred_x_y_1080[16][0])
            y1 = int(pred_x_y_1080[16][1])
            
            x2 = int(pred_x_y_1080[14][0])
            y2 = int(pred_x_y_1080[14][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
            
        if joints[18] and joints[18]:
            x1 = int(pred_x_y_1080[18][0])
            y1 = int(pred_x_y_1080[18][1])
            
            x2 = int(pred_x_y_1080[16][0])
            y2 = int(pred_x_y_1080[16][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
            
        
        #Rechtes Bein
        if joints[13] and joints[15]:
            x1 = int(pred_x_y_1080[15][0])
            y1 = int(pred_x_y_1080[15][1])
            
            x2 = int(pred_x_y_1080[13][0])
            y2 = int(pred_x_y_1080[13][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
            
        if joints[17] and joints[15]:
            x1 = int(pred_x_y_1080[17][0])
            y1 = int(pred_x_y_1080[17][1])
            
            x2 = int(pred_x_y_1080[15][0])
            y2 = int(pred_x_y_1080[15][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
            
        if joints[19] and joints[17]:
            x1 = int(pred_x_y_1080[19][0])
            y1 = int(pred_x_y_1080[19][1])
            
            x2 = int(pred_x_y_1080[17][0])
            y2 = int(pred_x_y_1080[17][1])
            
            img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
        img_array.append(img_out)
        cv2.imshow('Recording KINECT Video Stream', img_out)
        frame = None
        
        
    
    
    
    
    key = cv2.waitKey(1)
    if key == 27: 
        kinect.close()
        
        out = cv2.VideoWriter('merge_pose.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        
        cv2.destroyAllWindows()
        break