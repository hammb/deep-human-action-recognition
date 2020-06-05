import os
import sys
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

sys.path.append(os.path.join(os.getcwd(), 'benset')) 
from benset_batchloader_ar import *
from benset_dataloader_ar import *


num_frames = 8

dataset_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256"
num_action_predictions = 6
benset = Benset(dataset_path, num_action_predictions)

batch_size = 1

benset_seq = BatchLoader(benset, benset.get_train_data_keys(),benset.get_train_annotations(),batch_size,num_frames)

x, y = benset_seq.__next__()