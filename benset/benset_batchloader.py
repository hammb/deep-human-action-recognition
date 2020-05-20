import numpy as np
import math as calc
from tensorflow.python.keras.utils import Sequence

import json
import os
import cv2

from PIL import Image
from deephar.utils import *
from deephar.config import ntu_dataconf

# Here, `x_set` is list of path to the Frames
# and `y_set` are the associated classes.

class BatchLoader(Sequence):

    def __init__(self, dataloader, x_set, y_set, batch_size=6, num_frames = 16):
        
            self.dataloader = dataloader
            self.dataset_structure = dataloader.get_dataset_structure()
            self.x = x_set
            self.y = y_set
            self.batch_size = batch_size
            self.num_frames = num_frames
            self.batch_index = 0

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
                for frame in frames_of_sequence:
                    
                    #Path of each Frame of this Sequence
                    final_path = "{}{}{}".format(path, os.sep, frame)
                
                    #load Image
                    #img = T(Image.open(final_path))
                    img = cv2.imread(final_path,1)
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #resize to 256x256
                    img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                    #normalize
                    img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                        
                        
                    #restore RGB
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                    #resize to 256x256
                    #img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                
                    #normalize
                    #img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                
                    sequence.append(img_256_norm)
                    i = i + 1
                
                sequence = np.expand_dims(sequence, axis=0)
                batch_x.append(sequence)
            
            
            
            self.set_frame_list(batch_x)
            
            batch_y = self.y
            
            return batch_x

    def set_frame_list(self, batch_x):
        
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
    
    def get_frame_list(self):
        return self.frame_list
    
    def __next__(self):
        
        if self.batch_index <= self.__len__():
            batch = self.__getitem__(self.batch_index)
        else:
            batch = None
            
        self.batch_index = self.batch_index + 1
         
        return batch