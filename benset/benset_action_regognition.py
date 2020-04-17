import os
import sys
 
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

num_frames = 16
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
benset = Benset('datasets/Benset')
benset_seq = BatchLoader(benset.get_dataset_structure(), benset.get_test_data(),[0,0],2,num_frames)

x_batch, y_batch = benset_seq.__getitem__(0)
frame_list_groups = benset_seq.get_frame_list()

action_labels = ['baseball_pitch', 'baseball_swing', 'bench_press', 'bowl',
       'clean_and_jerk', 'golf_swing', 'jump_rope', 'jumping_jacks',
       'pullup', 'pushup', 'situp', 'squat', 'strum_guitar',
       'tennis_forehand', 'tennis_serve']

pred = []
index = 0
for frame_lists in frame_list_groups:
    sets = []
    for i in range(len(frame_lists)):
        
        first_frame = frame_lists[i].start
        last_frame = frame_lists[i].stop
        
        input = x_batch[1][:,first_frame:last_frame,:,:,:]
        prediction = models[1].predict(input)
        
        for p in range(len(prediction)):
            print(action_labels[np.argmax(prediction[p][0])] + " - (" + str(prediction[p][0][np.argmax(prediction[p][0])]) + ")")
           
        
        print("")   
        sets.append(prediction)
        
    pred.append(sets)
    index = index + 1
    


            