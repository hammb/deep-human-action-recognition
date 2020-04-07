import numpy as np
import math
from tensorflow.python.keras.utils import Sequence

import os
import cv2

# Here, `x_set` is list of path to the Frames
# and `y_set` are the associated classes.

class BatchLoader(Sequence):

    def __init__(self, dataset_structure, x_set, y_set, batch_size=1):
            self.dataset_structure = dataset_structure
            self.x = x_set
            self.y = y_set
            self.batch_size = batch_size

    def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        
            #If last batch
            if (idx + 1) == self.__len__():
                #Is last batch smaler than normal batchsize? 
                if (len(self.x) % self.batch_size) == 0:
                    batch_size = self.batch_size
                else:
                    #Set new size of smaller last batch
                    batch_size = (len(self.x) % self.batch_size)
            
            else:
                batch_size = self.batch_size
            
            for sequence_id in batch_size:
                
                
            
            #Path to Clip
            path = self.x[idx * self.batch_size:(idx + 1) * self.batch_size][0]
            #Last four characters of path (e.g. 0004)
            idx = self.x[idx * self.batch_size:(idx + 1) * self.batch_size][0][-4:]
        
            frames_of_sequence = self.dataset_structure[idx]
            
            sequence = []
            
            for frame in frames_of_sequence:
                
                #Path of each Frame of this Sequence
                final_path = "{}{}{}".format(path, os.sep, frame)
                
                #load Image
                img = cv2.imread(final_path,1)
                
                #restore RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                #resize to 256x256
                img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                
                #normalize
                img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                
                sequence.append(img_256_norm)
                
            sequence = np.expand_dims(sequence, 0)
            
            batch_x = sequence
            batch_y = [0]

            return np.array([]), np.array(batch_y)

    