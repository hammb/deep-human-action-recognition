import numpy as np
import math as calc
from tensorflow.python.keras.utils import Sequence

import json
import os
import cv2
import random
import time
from numpy.random import default_rng

from PIL import Image
from deephar.utils import *
from deephar.config import ntu_dataconf

# Here, `x_set` is list of path to the Frames
# and `y_set` are the associated classes.

class BatchLoader(Sequence):

    def __init__(self, dataloader, x_set, y_set, batch_size=6, num_frames = 16, mode=0, random_hflip=0, random_brightness=0, random_channel_shift=0, random_zoom=0, random_subsampling=0, random_rot=0, random_blur=0):
        
            self.dataloader = dataloader
            self.dataset_structure = dataloader.get_dataset_structure()
            self.x = x_set
            self.y = y_set
            self.batch_size = batch_size
            self.num_frames = num_frames
            self.batch_index = 0
            self.frame_list_index = 0
            self.next = int(time.time())
            self.frame_list = []
            self.mode = mode
            self.random_hflip = random_hflip
            self.random_brightness = random_brightness
            self.random_channel_shift = random_channel_shift
            self.random_zoom = random_zoom
            self.random_subsampling = random_subsampling
            self.random_rot = random_rot
            self.random_blur = random_blur

    def __len__(self):
            return calc.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        
            
            batch_x = []
            
            #If last batch
            if (idx + 1) >= self.__len__():
                #Is last batch smaler than normal batchsize? 
                if (len(self.x) % self.batch_size) == 0:
                    batch_size = self.batch_size
                else:
                    #Set new size of smaller last batch
                    batch_size = (len(self.x) % self.batch_size)
            
            else:
                batch_size = self.batch_size
            
            
            for batch_item in range(batch_size):
                
                clip_name = self.x[idx * self.batch_size:(idx + 1) * self.batch_size][batch_item]
                
                #Path to Clip
                path = self.dataloader.get_dataset_path()
                path += "\\frames\\"
                path += clip_name
                
                frames_of_sequence = self.dataset_structure[clip_name]
            
                sequence = []
                i = 0
                if self.random_hflip:
                    random.seed(int(time.time()))
                    hflip = random.randint(0, 1)
                else:
                    hflip = 0
                    
                if self.random_brightness:
                    random.seed(int(time.time()))
                    random_brightness = random.randint(0, 1)
                else: 
                    random_brightness = 0
                    
                if self.random_channel_shift:
                    random.seed(int(time.time()))
                    random_channel_shift = random.randint(0, 1)
                else:
                    random_channel_shift = 0
                    
                if self.random_zoom:
                    random.seed(int(time.time()))
                    random_zoom = random.randint(0, 1)
                else:
                    random_zoom = 0
                    
                if self.random_subsampling:
                    random.seed(int(time.time()))
                    random_subsampling = random.randint(0, 1)
                    
                    random.seed(int(time.time()))
                    subsample_rate = random.randint(2, 20)
                    
                    subsample_counter = 0
                else:
                    random_subsampling = 0
                
                if self.random_rot:
                    random.seed(int(time.time()))
                    random_rot = random.randint(0, 1)
                else:
                    random_rot = 0
                
                if self.random_blur:
                    random.seed(int(time.time()))
                    random_blur = random.randint(0, 1)
                else:
                    random_blur = 0
                
                for frame in frames_of_sequence:
                    
                    if random_subsampling:
                        if subsample_counter == subsample_rate:
                            subsample_counter = 0
                            continue
                        
                        subsample_counter = subsample_counter + 1
                    
                    #Path of each Frame of this Sequence
                    final_path = "{}{}{}".format(path, os.sep, frame)
                
                    #load Image
                    #img = T(Image.open(final_path))
                    img = cv2.imread(final_path,1)
                    
                    
                    if hflip:
                        
                        img = cv2.flip(img, 1)
                        
                    if random_brightness:
                        
                        random.seed(int(time.time()))
                        
                        if random.randint(0, 1):
                            value = random.uniform(0.3, 0.8)
                        else: 
                            value = random.uniform(1.2, 1.5)
                        
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        hsv = np.array(hsv, dtype = np.float64)
                        hsv[:,:,1] = hsv[:,:,1]*value
                        hsv[:,:,1][hsv[:,:,1]>255]  = 255
                        hsv[:,:,2] = hsv[:,:,2]*value 
                        hsv[:,:,2][hsv[:,:,2]>255]  = 255
                        hsv = np.array(hsv, dtype = np.uint8)
                        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        
                    if random_channel_shift:
                        
                        value = int(random.uniform(-60, 60))
                        img = img + value
                        img[:,:,:][img[:,:,:]>255]  = 255
                        img[:,:,:][img[:,:,:]<0]  = 0
                        img = img.astype(np.uint8)
                        
                        
                    if random_zoom:
                        
                        value = random.uniform(0.7, 1)
                        h, w = img.shape[:2]
                        h_taken = int(value*h)
                        w_taken = int(value*w)
                        h_start = random.randint(0, h-h_taken)
                        w_start = random.randint(0, w-w_taken)
                        img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
                        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
                        
                    if random_rot:
                        angle = int(random.uniform(-30, 30))
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
                        img = cv2.warpAffine(img, M, (w, h),borderValue=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)))
                        
                    if random_blur:
                        img = cv2.blur(img,(random.randint(2, 4),random.randint(2, 4)))
                      
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #resize to 256x256
                    img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                    
                
                    #normalize
                    #img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                        
                        
                    #restore RGB
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                    #resize to 256x256
                    #img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                
                    #normalize
                    #img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                
                    sequence.append(img_256)
                    i = i + 1
                
                sequence = np.expand_dims(sequence, axis=0)
                batch_x.append(sequence)
            
            
            
            self.set_frame_list(batch_x, self.mode)
            
            batch_y = self.y[clip_name]
            
            return batch_x, batch_y

    def set_frame_list(self, batch_x, mode=0): #mode 1 = train, mode 0 = test
        
        self.frame_list = []
        
        for batch in batch_x:
            
            num_frame_lists = int(len(batch[0]) / self.num_frames)
            
            sub_frame_list = []
            for i in range(num_frame_lists):
                start = (i) * self.num_frames
                end = (i+1) * self.num_frames
                sub_frame_list.append(range(start, end))
                
            if len(batch[0]) % self.num_frames != 0:
                
                sub_frame_list.append(range(len(batch[0])-1-self.num_frames, len(batch[0])-1))
                
            self.frame_list.append(sub_frame_list)
            
        if mode:
            random.seed(int(time.time()))
            new_index = random.randint(0,len(self.frame_list[0])-1)
            self.frame_list = [[self.frame_list[0][new_index]]]
    
    def get_frame_list(self):
        
        return self.frame_list
    
    def __next__(self):
        
        if self.next:
        
            if self.batch_index < self.__len__():
                self.sequence = self.__getitem__(self.batch_index)
            else:
                self.batch_index = 0
                self.frame_list_index = 0
                self.next = 1
                random.seed(int(time.time()))
                random.shuffle(self.x)
                return None, None
            
            self.frame_list_len = len(self.get_frame_list()[0])
            
            frame_list = self.get_frame_list()[0][self.frame_list_index]
        
            first_frame = frame_list.start
            last_frame = frame_list.stop
            self.frame_list_index = self.frame_list_index + 1    
            self.next = 0
            return self.sequence[0][0][:,first_frame:last_frame,:,:,:], self.sequence[1]
            
            
         
        else:
            
            if self.frame_list_index < self.frame_list_len:
                
                frame_list = self.get_frame_list()[0][self.frame_list_index]
        
                first_frame = frame_list.start
                last_frame = frame_list.stop
                self.frame_list_index = self.frame_list_index + 1    
                self.next = 0
                return self.sequence[0][0][:,first_frame:last_frame,:,:,:], self.sequence[1]
                
                
                
            else: 
                
                self.batch_index = self.batch_index + 1
                self.frame_list_index = 0
                
                
                if self.batch_index < self.__len__():
                    self.sequence = self.__getitem__(self.batch_index)
                else:
                    self.batch_index = 0
                    self.frame_list_index = 0
                    self.next = 1
                    random.seed(int(time.time()))
                    random.shuffle(self.x)
                    return None, None
                
                self.frame_list_len = len(self.get_frame_list()[0])
                
                frame_list = self.get_frame_list()[0][self.frame_list_index]
            
                first_frame = frame_list.start
                last_frame = frame_list.stop
                self.frame_list_index = self.frame_list_index + 1    
                self.next = 0
                return self.sequence[0][0][:,first_frame:last_frame,:,:,:], self.sequence[1]
                
    