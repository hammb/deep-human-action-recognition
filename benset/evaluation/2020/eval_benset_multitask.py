import os
import sys
import time
#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

from sklearn.metrics import confusion_matrix

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
         "E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0601\\weights_3dp+benset_ar_020.hdf5",
         by_name=True)

"""Save model callback."""

sys.path.append(os.path.join(os.getcwd(), 'benset')) 
from benset_batchloader_ar import *
from benset_dataloader_ar import *


dataset_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256_testdataset"
#dataset_path="D:\\sortout_Benset"
num_action_predictions = 6
benset = Benset(dataset_path, num_action_predictions)


batch_size = 1

benset_test = BatchLoader(benset, benset.get_dataset_keys(),benset.get_dataset_annotations(),batch_size, num_frames)
#benset_seq = BatchLoader(benset, benset.get_dataset_keys(),benset.get_dataset_annotations(),batch_size,num_frames, mode=1, random_hflip=1, random_brightness=1, random_channel_shift=0, random_zoom=1, random_subsampling=1, random_rot=1, random_blur=1)

from sklearn.metrics import confusion_matrix
y_actu = []
y_pred = []
from collections import Counter


predictions = []

wrong = [0,0,0,0]

while True:
    
    x, y = benset_test.__next__()
    
    if x is None:
        break
    
    prediction = full_model.predict(x)
    
    pred_action = np.argmax(prediction[11])
    annot_action = np.argmax(y[0])
    
    y_actu.append(annot_action)
    y_pred.append(pred_action)
    
    if pred_action == annot_action:
        predictions.append(1)
    else:
        wrong[annot_action] = wrong[annot_action] + 1
        predictions.append(0)
        
    accuracy = 100.0 / len(predictions) * Counter(predictions)[1]
    print("---")
    print(accuracy)
    print("---")

conf_mat = confusion_matrix(y_actu, y_pred)


#------------------------------------------------------------------------------------------------
predictions = []

wrong = [0,0,0,0]

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (0,0,0)
lineType               = 2

action_labels = ['Spritze', 'Anziehen', 'Ausziehen', 'Nichts']


while True:
    
    x, y = benset_test.__next__()
    
    if x is None:
        break
    
    prediction = full_model.predict(x)
    
    img = x[0][0]
    
    img = np.interp(img, (-1, 1), (0, 255))
    img = np.uint8(img)
    
    pred_action = np.argmax(prediction[11])
    annot_action = np.argmax(y[0])
    
    if pred_action == annot_action:
        cv2.putText(img, "RICHTIG", 
                    (100,200), 
                    font, 
                    fontScale,
                    (70, 168, 50),
                    lineType)
        predictions.append(1)
    else:
        wrong[annot_action] = wrong[annot_action] + 1
        cv2.putText(img, "FALSCH", 
                    (100,200), 
                    font, 
                    fontScale,
                    (214, 19, 19),
                    lineType)
        predictions.append(0)
    
    action = action_labels[np.argmax(prediction[11])]
    cv2.putText(img, str(action), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
    
    
    
    
    
    cv2.imshow('Recording KINECT Video Stream', img)
    key = cv2.waitKey(1)
    if key == 27: 
        
        cv2.destroyAllWindows()
        
        break
    #time.sleep(1)
    
        
    accuracy = 100.0 / len(predictions) * Counter(predictions)[1]
    print("---")
    print(accuracy)
    print("---")
    
cv2.destroyAllWindows()



