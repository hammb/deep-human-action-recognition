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
     
from benset_batchloader_ar import *
from benset_dataloader_ar import *


dataset_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256"
num_action_predictions = 6
benset = Benset(dataset_path, num_action_predictions)

dataset_list = benset.get_dataset_keys()

i = int(time.time())

sequences = []

while len(sequences) != 4640:

    random.seed(i)
    action = random.randint(0,3)
    i = i + 1
    
    random.seed(i)
    sequence = random.randint(0,799)
    i = i + 1
    
    random.seed(i)
    cam = random.randint(0,1)
    i = i + 1
    
    if sequence > 699 and action != 3:
        continue
    
    directory = "S%05dC%05dA%05d" % (sequence, cam, action)
    sequences.append(directory)
    sequences = list(dict.fromkeys(sequences))
    
sequences = sorted(sequences)

benset = Benset(dataset_path, num_action_predictions)
dataset_list = benset.get_dataset_keys()

for key in sequences:
    dataset_list.remove(key)

countA0 = []
countA1 = []
countA2 = []
countA3 = []

for sequence in sequences:
    if sequence.find("A00000") == 12:
        countA0.append(sequence)
    if sequence.find("A00001") == 12:
        countA1.append(sequence)
    if sequence.find("A00002") == 12:
        countA2.append(sequence)
    if sequence.find("A00003") == 12:
        countA3.append(sequence)


with open("C:\\networks\\deephar\\output\\split_train_val_benset\\train_list.p", 'wb') as fp:
    pickle.dump(sequences, fp, protocol=pickle.HIGHEST_PROTOCOL)
       
    
with open("C:\\networks\\deephar\\output\\split_train_val_benset\\val_list.p", 'wb') as fp:
    pickle.dump(dataset_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
    


"""
predictions = {}

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses.p", 'rb') as fp:
    predictions.update({0:pickle.load(fp)})

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses_prob.p", 'rb') as fp:
    predictions.update({1:pickle.load(fp)})
    
with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\pose_predictons.p", 'wb') as fp:
    pickle.dump(predictions, fp, protocol=pickle.HIGHEST_PROTOCOL)
"""
