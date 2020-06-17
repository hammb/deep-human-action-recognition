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
import collections 

sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader_benset import *
from benset_dataloader_ar import *


dataset_path_green="E:\\Bachelorarbeit-SS20\\datasets\\Benset256_green"


num_action_predictions = 6
use_backgrounds = True


benset_green = Benset(dataset_path_green, num_action_predictions,use_backgrounds=use_backgrounds)

batch_size = 1
num_frames = 8
mode=1
green_screen = 1

benset_seq = BatchLoader(dataloader=benset_green, x_set=benset_green.get_train_data_keys(),
                         y_set=benset_green.get_train_annotations(),batch_size=batch_size,
                         num_frames=num_frames,mode=mode,green_screen=green_screen, backgrounds=benset_green.get_backgrounds())

x, y = benset_seq.__next__()



frame_index = 0
for frame in x[0]:
    
    cv2.imshow('image',frame)
    cv2.waitKey(0)
    frame_index = frame_index + 1

test = []
index = 0
for frame in x:
    blank_image = np.zeros((256,256,3), np.uint8)
    blank_image[:,:] = (79, 225, 7)

    frame = cv2.resize(frame, (144,144), interpolation = cv2.INTER_AREA)

    x_offset=y_offset=56
    blank_image[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
    
    sequence[index] = blank_image
    
    index = index + 1


import cv2
import random
img = cv2.imread('arc_de_triomphe.jpeg')

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img

img = zoom(img, 1.5)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.resize(img, (100,100), interpolation = cv2.INTER_AREA)





dataset_structure = {}
counter = 0
dataset_path_end = "E:\\Bachelorarbeit-SS20\\datasets\\Benset256_green\\backgrounds\\"

for root, dirs, files in os.walk(os.path.join(os.getcwd(), dataset_path)):
    
    #Get names of sequences
    if dirs != []:
        if len(dirs) > 1:
            seq_structure = dirs
             
         
    #Get names of frames
    if files != []:
        if len(files) > 1:
             
        #Mapping of seqences and corresponding frames
            dataset_structure[seq_structure[counter]] = files
            counter += 1
            
dataset_path_256 = "E:\\Bachelorarbeit-SS20\\datasets\\benset256_backgrounds\\frames\\256\\"
dataset_path_square = "E:\\Bachelorarbeit-SS20\\datasets\\benset256_backgrounds\\frames\\squares\\"
frame_index = 104
for file in files:
    
    img = cv2.imread(dataset_path+"\\"+file,1)
    
    height, width, channels = img.shape
    
    if height < width:
        crop_img1 = img[0:height, 0:height]
        crop_img2 = img[(width-height):width, 0:height]
    else:
        crop_img1 = img[0:width, 0:width]
        crop_img2 = img[(height-width):height, 0:width]
    
    try:
        crop_img1 = cv2.resize(crop_img1, (256,256), interpolation = cv2.INTER_AREA)
        cv2.imwrite(dataset_path_end + "%05d.jpg" % (frame_index), crop_img1)
        frame_index = frame_index + 1
    except:
        print("err")
        
    try:
        crop_img2 = cv2.resize(crop_img2, (256,256), interpolation = cv2.INTER_AREA)
        cv2.imwrite(dataset_path_end + "%05d.jpg" % (frame_index), crop_img2)
        frame_index = frame_index + 1
    except:
        print("err")
    
    
    