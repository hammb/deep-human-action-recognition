import os
import sys
import numpy as np
import cv2
from PIL import Image
import time
 
sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from deephar.config import ModelConfig
from deephar.config import pennaction_dataconf
dconf = pennaction_dataconf.get_fixed_config()

from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

"""Load Model"""

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 32
cfg = ModelConfig((num_frames,) + pennaction_dataconf.input_shape, pa16j2d,
        num_actions=[15], num_pyramids=6, action_pyramids=[5, 6],
        num_levels=4, pose_replica=True,
        num_pose_features=160, num_visual_features=160)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)
        
"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(
        'output/weights_AR_merge_ep074_26-10-17.h5',
        by_name=True)

"""This call splits the model into its parts: pose estimation and action
recognition, so we can evaluate each part separately on its respective datasets.
"""
models = split_model(full_model, cfg, interlaced=False,
        model_names=['2DPose', '2DAction'])

"""Load kinect"""

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

"""Start capturing and prediction"""
input_data = []

action_labels = ['baseball_pitch', 'baseball_swing', 'bench_press', 'bowl',
       'clean_and_jerk', 'golf_swing', 'jump_rope', 'jumping_jacks',
       'pullup', 'pushup', 'situp', 'squat', 'strum_guitar',
       'tennis_forehand', 'tennis_serve']

frames = np.zeros((num_frames,) + pennaction_dataconf.input_shape)
frame_counter = 0
while True: 
    
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        #start_time = time.time() # start time of the loop
        
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
        
        img = T(Image.fromarray(framefullcolour))
        
        w, h = (img.size[0], img.size[1])
                    
        bbox = objposwin_to_bbox(np.array([w / 2, h / 2]), (dconf['scale']*max(w, h), dconf['scale']*max(w, h)))
                    
        objpos, winsize = bbox_to_objposwin(bbox)
              
        if min(winsize) < 32:
            winsize = (32, 32)
        
        objpos += dconf['scale'] * np.array([dconf['transx'], dconf['transy']])
        
        img.rotate_crop(dconf['angle'], objpos, winsize)
        img.resize(pennaction_dataconf.crop_resolution)
        img.normalize_affinemap()
                    
        frames[frame_counter-1, :, :, :] = normalize_channels(img.asarray(), channel_power=dconf['chpower'])
        
        #resize to 256x256
        #img_256 = cv2.resize(framefullcolour, (256,256), interpolation = cv2.INTER_AREA)
        #frames = np.squeeze(frames, axis=0)
        #normalize
        #img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
        #framefullcolour.show()
        #input_data.append(img)
        
        frame = None
        
        #print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    
    if frame_counter == num_frames:
        
        #frames = np.squeeze(frames, axis=0)
        #expand Dimension and Predict
        prediction = models[1].predict(np.expand_dims(frames, axis=0)) 
        print("--------------------------------")
        for p in prediction:
            print(action_labels[[i for i, x in enumerate(p[0]) if x == max(p[0])][0]])
            print("")
        print("--------------------------------")
        frames = np.zeros((num_frames,) + pennaction_dataconf.input_shape)
        frame_counter = 0
        input_data = []
    
    
    key = cv2.waitKey(1)
    if key == 27: 
        kinect.close()
        cv2.destroyAllWindows()
        break