import os
import sys
import time
import numpy as np
from PIL import Image
import pickle
import shutil
import cv2

sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader import *
from benset_dataloader import *

benset = Benset("E:\\Bachelorarbeit-SS20\\datasets\\Benset256")

dataset_structure = benset.get_dataset_structure()

predictions = {}

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses.p", 'rb') as fp:
    predictions.update({0:pickle.load(fp)})

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\poses_prob.p", 'rb') as fp:
    predictions.update({1:pickle.load(fp)})
    
sequneces = list(predictions[1].keys())

not_all_joints = []
sequences_with_not_all_joints = {} #joints missing on picture

A = [0,0,0,0]
B = [0,0,0,0]
C = [0,0,0,0]
D = [0,0,0,0]

for sequnece in sequneces:
    frames = predictions[1][sequnece]
    index = 0
    for frame in frames:
        res = [int(np.round(i, 0)) for i in frame]
        if res != [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]:
            if not sum(res) > 16:#too many points detected
                not_all_joints.append("%05d.jpg" % index)
        predictions[1][sequnece][index] = res
        index = index + 1
    if not_all_joints != []:
        sequences_with_not_all_joints.update({sequnece:not_all_joints})
    not_all_joints = []
    #predictions[1][sequnece] = res
   
"""     
Delete Frames from Sequences if only max 4 Frames wrong
Delete Sequences if more than 4 Frames wrong
"""
for i in sequences_with_not_all_joints:
    if len(sequences_with_not_all_joints[i])<3:
        for frames in sequences_with_not_all_joints[i]:
            if frames in dataset_structure[i]:
                dataset_structure[i].remove(frames)
    else:
        if i in dataset_structure:
            
            if i.find("A00000") == 12:
                A[0] = A[0] + 1
            if i.find("A00001") == 12:
                A[1] = A[1] + 1
            if i.find("A00002") == 12:
                A[2] = A[2] + 1
            if i.find("A00003") == 12:
                A[3] = A[3] + 1
            
            del dataset_structure[i]
        

"""
Delete Sequences if White Pants on
Action 1: 200 - 399
Action 2:   0 - 199
"""

for i in range(200,400):
    if "S%05dC%05dA%05d" % (i, 0, 1) in dataset_structure:
        del dataset_structure["S%05dC%05dA%05d" % (i, 0, 1)]
    if "S%05dC%05dA%05d" % (i, 1, 1) in dataset_structure:
        del dataset_structure["S%05dC%05dA%05d" % (i, 1, 1)]
B = [0,200,0,0]    
for i in range(0,200):
    if "S%05dC%05dA%05d" % (i, 0, 2) in dataset_structure:
        del dataset_structure["S%05dC%05dA%05d" % (i, 0, 2)]
    if "S%05dC%05dA%05d" % (i, 1, 2) in dataset_structure:
        del dataset_structure["S%05dC%05dA%05d" % (i, 1, 2)]
C = [0,0,200,0]      
    

"""
Abstaende Hüfte/Knie/Füße
"""

all_sequences_with_mistake = {}
with_mistake = []

for sequence in dataset_structure:
    index = 0
    for frame_index in range(len(dataset_structure[sequence])):
    
        #Links (Aufs Bild schauend)
        joint12 = predictions[0][sequence][frame_index][12][0:2]
        joint14 = predictions[0][sequence][frame_index][14][0:2]
        joint16 = predictions[0][sequence][frame_index][16][0:2]
        
        #Rechts
        joint13 = predictions[0][sequence][frame_index][13][0:2]
        joint15 = predictions[0][sequence][frame_index][15][0:2]
        joint17 = predictions[0][sequence][frame_index][17][0:2]
        
        """
        überkreuzte Beine
        """
        """
        0 0
         X        
        0 0
         X
        0 0
        """
        if(joint12[0] < joint13[0]):
            if(joint16[0] < joint17[0]):
                if not (joint14[0] > joint15[0]):
                    if not np.absolute(joint12[0] - joint13[0]) < 0.1:
                        with_mistake.append("%05d.jpg" % index)
                        index = index + 1
                        continue
            
        if(joint12[0] > joint13[0]):
            if(joint16[0] > joint17[0]):
                if not (joint14[0] < joint15[0]):
                    if not np.absolute(joint12[0] - joint13[0]) < 0.1:
                        with_mistake.append("%05d.jpg" % index)
                        index = index + 1
                        continue
        
        """
        0 0
        | |       
        0 0
         X
        0 0
        """
        if(joint12[0] < joint13[0]):
            if(joint14[0] < joint15[0]):
                if not (joint16[0] > joint17[0]):
                    if not np.absolute(joint12[0] - joint13[0]) < 0.1:
                        with_mistake.append("%05d.jpg" % index)
                        index = index + 1
                        continue
            
        if(joint12[0] > joint13[0]):
            if(joint14[0] > joint15[0]):
                if not (joint16[0] < joint17[0]):
                    if not np.absolute(joint12[0] - joint13[0]) < 0.1:
                        with_mistake.append("%05d.jpg" % index)
                        index = index + 1
                        continue
        """
        0 0
         X       
        0 0
        | |
        0 0
        """
        if(joint16[0] < joint17[0]):
            if(joint14[0] < joint15[0]):
                if not (joint12[0] > joint13[0]):
                    if not np.absolute(joint12[0] - joint13[0]) < 0.1:
                        with_mistake.append("%05d.jpg" % index)
                        index = index + 1
                        continue
            
        if(joint16[0] > joint17[0]):
            if(joint14[0] > joint15[0]):
                if not (joint12[0] < joint13[0]):
                    if not np.absolute(joint12[0] - joint13[0]) < 0.1:
                        with_mistake.append("%05d.jpg" % index)
                        index = index + 1
                        continue
        
        lenght_12_14 = np.linalg.norm(joint12-joint14)
        lenght_14_16 = np.linalg.norm(joint14-joint16)
        
        lenght_13_15 = np.linalg.norm(joint13-joint15)
        lenght_15_17 = np.linalg.norm(joint15-joint17)
        
        lenghts = [lenght_12_14, lenght_14_16, lenght_13_15, lenght_15_17]
        
        #get medium high value (not min or max)
        
        indeces_of_lenghts = [0, 1, 2, 3]
        
        index_min = lenghts.index(min(lenghts))
        index_max = lenghts.index(max(lenghts))
        
        #remove min and max value from indeces list
        indeces_of_lenghts.remove(index_min)
        indeces_of_lenghts.remove(index_max)
        
        medium_high_value = lenghts[indeces_of_lenghts[0]]
        
        #calc 25% tolerance
        tolerance = medium_high_value * 0.5
        
        #compare
        diff = []
        
        diff.append(np.absolute(lenght_12_14 - lenght_14_16))
        diff.append(np.absolute(lenght_12_14 - lenght_14_16))
        diff.append(np.absolute(lenght_12_14 - lenght_14_16))
        diff.append(np.absolute(lenght_12_14 - lenght_14_16))
        
        longest_difference = max(diff)
    
        if(longest_difference > tolerance):
            with_mistake.append("%05d.jpg" % index)
            
        index = index + 1
    if with_mistake != []:
        all_sequences_with_mistake.update({sequence:with_mistake})
    with_mistake = []


for i in all_sequences_with_mistake:
    if len(all_sequences_with_mistake[i])<3:
       for frames in all_sequences_with_mistake[i]:
            if frames in dataset_structure[i]:
                dataset_structure[i].remove(frames)
    else:
        if i in dataset_structure:
            
            if i.find("A00000") == 12:
                D[0] = D[0] + 1
            if i.find("A00001") == 12:
                D[1] = D[1] + 1
            if i.find("A00002") == 12:
                D[2] = D[2] + 1
            if i.find("A00003") == 12:
                D[3] = D[3] + 1
            
            del dataset_structure[i]

    
countA0 = 0
countA1 = 0
countA2 = 0
countA3 = 0
countC0 = 0
countC1 = 0
for sequence in dataset_structure:
    if sequence.find("A00000") == 12:
        countA0 = countA0 + 1
    if sequence.find("A00001") == 12:
        countA1 = countA1 + 1
    if sequence.find("A00002") == 12:
        countA2 = countA2 + 1
    if sequence.find("A00003") == 12:
        countA3 = countA3 + 1
    if sequence.find("C00000") == 6:
        countC0 = countC0 + 1
    if sequence.find("C00001") == 6:
        countC1 = countC1 + 1   

with open("E:\\Bachelorarbeit-SS20\\datasets\\Benset256Test\\updated_clean_dataset_structure.p", 'wb') as fp:
    pickle.dump(dataset_structure, fp, protocol=pickle.HIGHEST_PROTOCOL)
