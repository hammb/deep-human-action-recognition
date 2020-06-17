import numpy as np
import math as calc
from tensorflow.python.keras.utils import Sequence

import json
import os
import cv2
import random
from numpy.random import default_rng

from PIL import Image
from deephar.utils import *
from deephar.config import ntu_dataconf

import imgaug as ia
from imgaug import augmenters as iaa

import time

# Here, `x_set` is list of path to the Frames
# and `y_set` are the associated classes.

class BatchLoader(Sequence):

    def __init__(self, dataloader, x_set, y_set, batch_size=6, num_frames = 16, mode=0, random_subsampling=0, green_screen=0, backgrounds=None):
        
            self.dataloader = dataloader
            self.dataset_structure = dataloader.get_dataset_structure()
            self.x = x_set
            self.y = y_set
            self.batch_size = batch_size
            self.num_frames = num_frames
            self.batch_index = 0
            self.frame_list_index = 0
            self.next = 1
            self.temp_dataset_list = x_set
            self.frame_list = []
            self.mode = mode
            self.random_subsampling = random_subsampling
            self.green_screen = green_screen
            self.seed = int(time.time())
            self.backgrounds = backgrounds
            

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
                
                if self.random_subsampling:
                    
                    random.seed(self.seed)
                    self.inc_seed()
                    
                    random_subsampling = random.randint(0, 1)
                    
                    random.seed(self.seed)
                    self.inc_seed()
                    
                    subsample_rate = random.randint(2, 20)
                    
                    subsample_counter = 0
                else:
                    random_subsampling = 0
                
                
                seed_count = 1
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
                    
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #resize to 256x256
                    img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                    
                    
                    #normalize
                    
                        
                    #restore RGB
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                    #resize to 256x256
                    #img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                
                    #normalize
                    #img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                    
                    sequence.append(img_256)
                    
                    
                    i = i + 1
                
                
                
                
                if self.mode:
                    sequence = self.data_augmentation(sequence)
                   
                frame_index = 0
                for frame in sequence:
                    
                    #sequence[frame_index] = cv2.normalize(frame, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                    frame_index = frame_index + 1
                
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
            
            random.seed(self.seed)
            self.inc_seed()
            
            new_index = int(random.gauss((len(self.frame_list[0])-1)//2, (len(self.frame_list[0])-1)//5))
                            
            if new_index > len(self.frame_list[0])-1:
                new_index = len(self.frame_list[0])-1
                
            if new_index < 0:
                new_index = 0
            
            #new_index = random.randint(0,len(self.frame_list[0])-1)
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
                
                random.seed(self.seed)
                self.inc_seed()
                
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
                    random.seed(self.seed)
                    self.inc_seed()
                    random.shuffle(self.x)
                    return None, None
                
                self.frame_list_len = len(self.get_frame_list()[0])
                
                frame_list = self.get_frame_list()[0][self.frame_list_index]
            
                first_frame = frame_list.start
                last_frame = frame_list.stop
                self.frame_list_index = self.frame_list_index + 1    
                self.next = 0
                return self.sequence[0][0][:,first_frame:last_frame,:,:,:], self.sequence[1]
        
    def inc_seed(self):
        
        try:
            self.seed = self.seed + random.randint(1,2)
        except:
            self.seed = 1
        
    def data_augmentation(self, sequence):
        
        """
        square or 16x9
        
        """
        random.seed(self.seed)
        self.inc_seed() 
        
        square = random.randint(0,1)
        """
        hflip
        
        """
        random.seed(self.seed)
        self.inc_seed() 
        
        hflip = random.randint(0,1)
        
        hflip_aug = iaa.Sequential([iaa.Fliplr(1)])
        hflip_det = hflip_aug.to_deterministic()
        
        zoom_scale_rot_aug = iaa.Sequential([
                                    iaa.Affine(
                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                    rotate=(-25, 25),
                                    shear=(-8, 8)
                                    )], random_order=True)
    
        zoom_scale_rot_det = zoom_scale_rot_aug.to_deterministic()
        
        counter = 0
        for frame in sequence:
            
            if square:
                """
                square or 16x9
                
                """
                blank_image = np.zeros((256,256,3), np.uint8)
                blank_image[:,:] = (79, 225, 7)
    
                frame = cv2.resize(frame, (144,144), interpolation = cv2.INTER_AREA)
    
                x_offset=y_offset=56
                blank_image[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
                
                frame = blank_image
            
            if hflip:
                """
                hflip
                
                """
                frame = hflip_det.augment_image(frame)
            
            
            """
            Zoom & Verschiebung
            """
            
            frame = zoom_scale_rot_det.augment_image(frame)
            
            """
            black Background after rotation to green
            
            """
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        
            #sensitivity is a int, typically set to 15 - 20 

            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 10])
            
            mask = cv2.inRange(hsv, lower_black, upper_black)
            
            frame[mask>0]=(79,255,7)
            
            
            """
            Background Pic or Color
            
            """
            
            if self.green_screen:
                        
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                #sensitivity is a int, typically set to 15 - 20 
    
                sensitivity = 30
                
                lower_green = np.array([70 - sensitivity, 100, 100])
                upper_green = np.array([70 + sensitivity, 255, 255])
                
                mask = cv2.inRange(hsv, lower_green, upper_green)
                
                if self.backgrounds is None:
                    random.seed(self.seed)
                    self.inc_seed()
                    
                    r = random.randint(0, 255)
                    
                    
                    random.seed(self.seed)
                    self.inc_seed()
                    
                    g = random.randint(0, 255)
                    
                    
                    random.seed(self.seed)
                    self.inc_seed()
                    
                    b = random.randint(0, 255)
                    
                    frame[mask>0]=(r,g,b)
                else:
                    
                    frame[mask != 0] = [0, 0, 0]
                    
                    random.seed(self.seed)
                    self.inc_seed()
                    
                    back_path = self.dataloader.get_dataset_path()
                    back_path += "\\backgrounds\\"
                    back_path += self.backgrounds[random.randint(0, len(self.backgrounds) - 1)]
                    
                    
                    crop_background = cv2.imread(back_path, 1)
                    
                    random.seed(self.seed)
                    self.inc_seed()
                    
                    crop_background = cv2.flip(crop_background, random.randint(-1, 1))
                    
                    crop_background[mask == 0] = [0, 0, 0]
                    
                    frame = crop_background + frame
            
            
            sequence[counter] = frame
            counter = counter + 1
        
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        
        seq = iaa.Sequential(
                                [
                                    
                                   
                            
                                   
                            
                                    #
                                    # Execute 0 to 5 of the following (less important) augmenters per
                                    # image. Don't execute all of them, as that would often be way too
                                    # strong.
                                    #
                                    iaa.SomeOf((0, 3),
                                        [
                                            # Convert some images into their superpixel representation,
                                            # sample between 20 and 200 superpixels per image, but do
                                            # not replace all superpixels with their average, only
                                            # some of them (p_replace).
                                            sometimes(
                                                iaa.Superpixels(
                                                    p_replace=(0, 1.0),
                                                    n_segments=(20, 200)
                                                )
                                            ),
                            
                                            # Blur each image with varying strength using
                                            # gaussian blur (sigma between 0 and 3.0),
                                            # average/uniform blur (kernel size between 2x2 and 7x7)
                                            # median blur (kernel size between 3x3 and 11x11).
                                            iaa.OneOf([
                                                iaa.GaussianBlur((0, 3.0)),
                                                iaa.AverageBlur(k=(2, 7)),
                                                iaa.MedianBlur(k=(3, 11)),
                                            ]),
                            
                                            # Sharpen each image, overlay the result with the original
                                            # image using an alpha between 0 (no sharpening) and 1
                                            # (full sharpening effect).
                                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                            
                                            # Same as sharpen, but for an embossing effect.
                                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                            
                                            # Search in some images either for all edges or for
                                            # directed edges. These edges are then marked in a black
                                            # and white image and overlayed with the original image
                                            # using an alpha of 0 to 0.7.
                                            sometimes(iaa.OneOf([
                                                iaa.EdgeDetect(alpha=(0, 0.7)),
                                                iaa.DirectedEdgeDetect(
                                                    alpha=(0, 0.7), direction=(0.0, 1.0)
                                                ),
                                            ])),
                            
                                            # Add gaussian noise to some images.
                                            # In 50% of these cases, the noise is randomly sampled per
                                            # channel and pixel.
                                            # In the other 50% of all cases it is sampled once per
                                            # pixel (i.e. brightness change).
                                            iaa.AdditiveGaussianNoise(
                                                loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                                            ),
                            
                                            # Either drop randomly 1 to 10% of all pixels (i.e. set
                                            # them to black) or drop them on an image with 2-5% percent
                                            # of the original size, leading to large dropped
                                            # rectangles.
                                            iaa.OneOf([
                                                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                                iaa.CoarseDropout(
                                                    (0.03, 0.15), size_percent=(0.02, 0.05),
                                                    per_channel=0.2
                                                ),
                                            ]),
                            
                                        
                                            # Change brightness of images (50-150% of original value).
                                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            
                                            # Improve or worsen the contrast of images.
                                            iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                            
                                            # Convert each image to grayscale and then overlay the
                                            # result with the original with random alpha. I.e. remove
                                            # colors with varying strengths.
                                            iaa.Grayscale(alpha=(0.0, 1.0)),
                            
                                           
                            
                                            # In some images distort local areas with varying strength.
                                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                                        ],
                                        # do all of the above augmentations in random order
                                        random_order=True
                                    )
                                ],
                                # do all of the above augmentations in random order
                                random_order=True
                            )
        sequence = seq(images=sequence)
        return sequence
  
    