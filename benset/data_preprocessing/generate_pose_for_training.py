import os
import sys
import time
import numpy as np
from PIL import Image
import pickle

#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())

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
#import annothelper

#annothelper.check_ntu_dataset()

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

def save_test_sample(input,prediction, sequences, i):
    
    
    img = input[0][10]
    #img = cv2.resize(img, (500,500), interpolation = cv2.INTER_AREA)
    img = np.interp(img, (-1, 1), (0, 255))
    img = np.uint16(img)
            
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    pred_x_y_z_1 = prediction[6][i]

    pred_x_y_1 = pred_x_y_z_1[:,0:2]
    pred_x_y_1080 = np.interp(pred_x_y_1, (0, 1), (0, 256))
    
    joints = []
    for x in prediction[7][i]:
        if x[0] > 0.5:
            joints.append(1)
        else:
            joints.append(0)
         
    index = 0
    for x in pred_x_y_1080:
        if prediction[7][i][index][0] > 0.5:  
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
        
    if joints[18] and joints[16]:
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
        
    
    cv2.imwrite("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\" + sequences + "F%05d.jpg" % i, img_out)
    
    


"""Load Benset dataset."""
benset = Benset("E:\\Bachelorarbeit-SS20\\datasets\\Benset256")


benset_seq = BatchLoader(benset, benset.get_dataset(),[0,0],20,num_frames)

x_batch = benset_seq.__next__()

sequences = benset.get_dataset()
dataset_index = 0 
poses_of_sequence = []
joint_probs = []
all_poses = {}
all_joint_probs = {}

run = True
while run:
    
    frame_list_groups = benset_seq.get_frame_list()
    
    index = 0
    
    for frame_lists in frame_list_groups:
        
        last_range_size = (frame_lists[-1].stop - frame_lists[-2].stop) + 1
        
        for i in range(len(frame_lists)):
            
            first_frame = frame_lists[i].start
            last_frame = frame_lists[i].stop
            
            input = x_batch[index][:,first_frame:last_frame,:,:,:]
            prediction = model_pe.predict(input[0])
            if i == (range(len(frame_lists)).stop - 1):
                if len(x_batch[index][0]) % 20 == 0:
                    poses_of_sequence.extend(prediction[6])
                    joint_probs.extend(prediction[7])
                else:
                    poses_of_sequence.extend(prediction[6][-last_range_size:])
                    joint_probs.extend(prediction[7][-last_range_size:])
            else:
                poses_of_sequence.extend(prediction[6])
                joint_probs.extend(prediction[7])
                
            
            #save_test_sample(input,prediction,sequences[dataset_index],i)
            
        all_poses.update({sequences[dataset_index]:poses_of_sequence})
        all_joint_probs.update({sequences[dataset_index]:joint_probs})
        
        poses_of_sequence = []
        joint_probs = []
        dataset_index = dataset_index + 1
        print(dataset_index)
        index = index + 1
    
    x_batch = benset_seq.__next__()
    
    if x_batch == None:
        run = False

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses.p", 'wb') as fp:
    pickle.dump(all_poses, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses_prob.p", 'wb') as fp:
    pickle.dump(all_joint_probs, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
"""
with open("E:\\Bachelorarbeit-SS20\\datasets\\TestBenset\\poses.p", 'rb') as fp:
    poses = pickle.load(fp)
"""