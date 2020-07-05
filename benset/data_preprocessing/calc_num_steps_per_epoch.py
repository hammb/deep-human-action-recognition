import os
import sys

num_frames = 6

batch_clips = 1 # 8/4


sys.path.append(os.path.join(os.getcwd(), 'benset')) 
from benset_batchloader_ar import *
from benset_dataloader_ar import *


dataset_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256"
num_action_predictions = 6
benset = Benset(dataset_path, num_action_predictions)

num_frames = 6
batch_size = 1

benset_seq = BatchLoader(benset, benset.get_train_data_keys(),benset.get_train_annotations(),batch_size,num_frames)

count = 0
steps_per_epoch = 0
for _ in benset.get_train_data_keys():
    x,y = benset_seq.__getitem__(count)
    steps_per_epoch = steps_per_epoch + len(benset_seq.get_frame_list()[0])
    print(steps_per_epoch)
    count = count + 1


    
    
    