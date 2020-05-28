import os
import sys
import time
import numpy as np
from PIL import Image
import pickle
import shutil
import cv2
import random

sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader import *
from benset_dataloader import *

dataset_structure_file_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\updated_clean_dataset_structure.p"
dataset_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256"
adjust_dataset_structure=True

benset = Benset(dataset_path, dataset_structure_file_path, adjust_dataset_structure)

num_frames = 20
batch_size = 6

benset_seq = BatchLoader(benset, benset.get_train_data(),[0,0],batch_size,num_frames)

x_batch = benset_seq.__next__()


predictions = {}

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses.p", 'rb') as fp:
    predictions.update({0:pickle.load(fp)})

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses_prob.p", 'rb') as fp:
    predictions.update({1:pickle.load(fp)})
    

    
#Find Smallest Class, and randomly make all classes equal big
"""    
countA0 = []
countA1 = []
countA2 = []
countA3 = []

for sequence in dataset_structure:
    if sequence.find("A00000") == 12:
        countA0.append(sequence)
    if sequence.find("A00001") == 12:
        countA1.append(sequence)
    if sequence.find("A00002") == 12:
        countA2.append(sequence)
    if sequence.find("A00003") == 12:
        countA3.append(sequence)
        
smallest_class = min([len(countA0), len(countA1), len(countA2), len(countA3)])

for i in range(0,len(countA0)-smallest_class):
    index = random.randrange(0, len(countA0), 1)
    dataset_structure.pop(countA0[index])
    countA0.pop(index)
    
for i in range(0,len(countA1)-smallest_class):
    index = random.randrange(0, len(countA1), 1)
    dataset_structure.pop(countA1[index])
    countA1.pop(index)
    
for i in range(0,len(countA2)-smallest_class):
    index = random.randrange(0, len(countA2), 1)
    dataset_structure.pop(countA2[index])
    countA2.pop(index)
    
for i in range(0,len(countA3)-smallest_class):
    index = random.randrange(0, len(countA3), 1)
    dataset_structure.pop(countA3[index])
    countA3.pop(index)
    
"""