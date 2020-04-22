import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader import *
from benset_dataloader import *

from deephar.config import ModelConfig
from deephar.config import pennaction_dataconf

from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *
"""Load Model"""

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
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


"""Load Data"""
benset = Benset('datasets/BensetCroppedSmall')
benset_seq = BatchLoader(benset.get_dataset_structure(), benset.get_train_data(),[0,0],3,num_frames)

x_batch, y_batch = benset_seq.__getitem__(0)
frame_list_groups = benset_seq.get_frame_list()

action_labels = ['baseball_pitch', 'baseball_swing', 'bench_press', 'bowl',
       'clean_and_jerk', 'golf_swing', 'jump_rope', 'jumping_jacks',
       'pullup', 'pushup', 'situp', 'squat', 'strum_guitar',
       'tennis_forehand', 'tennis_serve']

all_blocks = np.zeros(shape=(54, 15))
pred = []
index = 0
anz_frames = 0
for frame_lists in frame_list_groups:
    print("---------------")
    print(index)
    sets = []
    
    for i in range(len(frame_lists)):
        
        first_frame = frame_lists[i].start
        last_frame = frame_lists[i].stop
        
        img = x_batch[index][:,first_frame,:,:,:][0]
        img = np.interp(img, (-1, 1), (0, 256))
        img = Image.fromarray(np.uint8(img))
        
        img.show()
        
        input = x_batch[index][:,first_frame:last_frame,:,:,:]
        prediction = models[1].predict(input)
        
        
        block_ind = 0
        for block in prediction:
            print("Blocknr : %d" % block_ind)
            
            print(action_labels[np.argmax(block)])
            
            block_in = 0
            for value in prediction[-1][0]:
                
                all_blocks[anz_frames, block_in] = value
                block_in = block_in + 1
            block_ind = block_ind +1
        print("-")
        sets.append(prediction)
        anz_frames = anz_frames + 1 
    pred.append(sets)
    
    index = index + 1
    
evaluated_predictions = []

for sequences in pred:
    for predictions in sequences:
        result = np.ones(15)
        for prediction_result in predictions:
            result *= prediction_result[0]
            
    evaluated_predictions.append(result)
    
for sequences in evaluated_predictions:
    print(action_labels[int(np.argmax(sequences))])
            