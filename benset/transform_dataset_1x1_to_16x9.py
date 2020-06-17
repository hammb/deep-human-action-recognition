import os
import sys
import time
import numpy as np
from PIL import Image
import pickle
import shutil
import cv2
import random
from collections import Counter

sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader_benset import *
from benset_dataloader_ar import *


dataset_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256_green"
num_action_predictions = 6
benset = Benset(dataset_path, num_action_predictions)

dataset_list = benset.get_dataset_keys()

batch_size = 1
num_frames = 8
mode=0

benset_seq = BatchLoader(dataloader=benset, x_set=benset.get_train_data_keys(),
                         y_set=benset.get_train_annotations(),batch_size=batch_size,
                         num_frames=num_frames,mode=mode)

x, y = benset_seq.__next__()



frame_index = 0
for frame in x[0]:
    blank_image = np.zeros((256,256,3), np.uint8)
    blank_image[:,:] = (79, 225, 7)
    
    frame = cv2.resize(frame, (144,144), interpolation = cv2.INTER_AREA)
    
    x_offset=y_offset=56
    blank_image[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
    
    cv2.imshow('image',blank_image)
    cv2.waitKey(0)
    frame_index = frame_index + 1


