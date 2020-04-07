import os
import cv2
import numpy as np
import magic
import re

class Benset(object):
    def __init__(self, dataset_path, annotation_file=None):
        self.dataset_path = dataset_path
        self.annotation_file = annotation_file
        
        if self.annotation_file is None:  
            self.generate_annotations()
        
    def get_dataset(self):
        
        sequence = {}
        sequence_preprocessed = {}
        #Names of Sequences (0001, 0002, ...)
        seq_structure_keys = list(self.seq_structure.keys())
        
        #Iterate over all sequences
        for seq_key in range(len(seq_structure_keys)):
            image = {}
            image_preprocessed = {}
            #Path of each Sequence
            path_dir = os.path.join(os.getcwd(), self.dataset_path)
            path_dir = "{}{}{}".format(path_dir, os.sep, 'frames')
            path_dir = "{}{}{}".format(path_dir, os.sep, seq_structure_keys[seq_key])
            
            #Frames of an indiviudal Sequence
            frames_of_seq = self.seq_structure[seq_structure_keys[seq_key]]
            
            #Iterate over frames in Sequence
            for frame_key in range(len(frames_of_seq)):
                
                
                #Get Filename of each Frame
                frame_name = frames_of_seq[frame_key]
                
                #Path of each Frame of this Sequence
                final_path = "{}{}{}".format(path_dir, os.sep, frame_name)
                
                #load Image
                img = cv2.imread(final_path,1)
                
                #restore RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                #resize to 256x256
                img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                #normalize
                img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                
                image_preprocessed[frame_key] = img_256_norm
                image[frame_key] = img
            
            sequence[seq_key] = image
            #expand Dimension
            sequence_preprocessed[seq_key] = np.expand_dims(image_preprocessed, 0)
            
        output = {}
        output['proportion_train_test'] = self.proportion_train_test
        output['images'] = sequence
        output['sequences_preprocessed'] = sequence_preprocessed
        output['seq_structure'] = self.seq_structure
        output['img_sizes'] = self.seq_sizes
        
        return output
        
    def generate_annotations(self):
        
        proportion_train_test = 4
        
        self.seq_structure = {}
        self.seq_sizes = {}
        self.proportion_train_test = {}
        counter = 0
        
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), self.dataset_path)):
           
            #Get names of sequences
            if dirs != []:
                if len(dirs) > 1:
                    seq_structure = dirs
                    
                
            #Get names of frames and picture sizes
            if files != []:
                if len(files) > 1:
                    
                    
                    #Division into test and training data
                    if counter%proportion_train_test == 0:
                        if 'test' in self.proportion_train_test:
                            self.proportion_train_test.setdefault('test', []).append(seq_structure[counter])
                        else: 
                            self.proportion_train_test['test'] = []
                            self.proportion_train_test.setdefault('test', []).append(seq_structure[counter])
                    else:
                        if 'train' in self.proportion_train_test:
                            self.proportion_train_test.setdefault('train', []).append(seq_structure[counter])
                        else: 
                            self.proportion_train_test['train'] = []
                            self.proportion_train_test.setdefault('train', []).append(seq_structure[counter])
                            
                            
                        
                        
                    #Mapping of seqences and corresponding frames
                    self.seq_structure[seq_structure[counter]] = files
                    
                    #Get filepath to first frame in sequence
                    path_dir = os.path.join(os.getcwd(), self.dataset_path)
                    path_file = "{}{}{}".format(path_dir, os.sep, 'frames')
                    path_file = "{}{}{}".format(path_file, os.sep, seq_structure[counter])
                    path_file = "{}{}{}".format(path_file, os.sep, files[0])
                    
                    #Load dimensions of image
                    seq_sizes = magic.from_file(path_file)
                    seq_sizes = re.search('(\d\d\d+)x(\d\d\d+)', seq_sizes).groups()
                    
                    self.seq_sizes[seq_structure[counter]] = seq_sizes
                    
                    counter += 1
                    
    def get_one_seqence_from_dataset(self, key):
        """
        key = index of sequence in Dataset
        
        """
        
        sequence = {}
        sequence_preprocessed = {}
        #Names of sequences (0001, 0002, ...)
        seq_structure_keys = list(self.seq_structure.keys())
        
        assert len(seq_structure_keys) > key, \
            "Key %d ist greater than listlength %d. (Key counts from 0 to listlength-1)" % (key, len(seq_structure_keys))
        
        seq_key = key 
        
        image = {}
        image_preprocessed = {}
        #Path of each sequence
        path_dir = os.path.join(os.getcwd(), self.dataset_path)
        path_dir = "{}{}{}".format(path_dir, os.sep, 'frames')
        path_dir = "{}{}{}".format(path_dir, os.sep, seq_structure_keys[seq_key])
            
        #Frames of an indiviudal sequence
        frames_of_seq = self.seq_structure[seq_structure_keys[seq_key]]
            
        #Iterate over frames in sequence
        for frame_key in range(len(frames_of_seq)):
                
                
            #Get filename of each frame
            frame_name = frames_of_seq[frame_key]
                
            #Path of each frame of this sequence
            final_path = "{}{}{}".format(path_dir, os.sep, frame_name)
                
            #load image
            img = cv2.imread(final_path,1)
                
            #restore RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            #resize to 256x256
            img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            #normalize
            img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                
            image_preprocessed[frame_key] = img_256_norm
            image[frame_key] = img
            
        sequence[seq_key] = image
        #expand dimension
        sequence_preprocessed[seq_key] = np.expand_dims(image_preprocessed, 0)
            
        output = {}
        output['proportion_train_test'] = self.proportion_train_test
        output['images'] = sequence
        output['sequences_preprocessed'] = sequence_preprocessed
        output['seq_structure'] = self.seq_structure[seq_structure_keys[seq_key]]
        output['img_sizes'] = self.seq_sizes[seq_structure_keys[seq_key]]
        
        return output
    
    def get_data_from_mode(self, mode):
        """
        mode = 'train' or 'test'
        """
        assert mode == 'train' or mode == 'test', \
            "Unknown mode"
        
        filter_seq_structure = {}
        filter_seq_sizes = {}
        
        sequence = {}
        sequence_preprocessed = {}
        #Names of Sequences (0001, 0002, ...)
        seq_structure_keys = list(self.proportion_train_test[mode])
        
        #Iterate over all sequences
        for seq_key in range(len(seq_structure_keys)):
            image = {}
            image_preprocessed = {}
            
            #Path of each Sequence
            path_dir = os.path.join(os.getcwd(), self.dataset_path)
            path_dir = "{}{}{}".format(path_dir, os.sep, 'frames')
            path_dir = "{}{}{}".format(path_dir, os.sep, seq_structure_keys[seq_key])
            
            #Frames of an indiviudal Sequence
            frames_of_seq = self.seq_structure[seq_structure_keys[seq_key]]
            
            
            #Sequence structures and image sizes of this part of the dataset
            if seq_structure_keys[seq_key] in filter_seq_structure:
                filter_seq_structure.setdefault(seq_structure_keys[seq_key], []).append(self.seq_structure[seq_structure_keys[seq_key]])
            else:
                filter_seq_structure[seq_structure_keys[seq_key]] = []
                filter_seq_structure.setdefault(seq_structure_keys[seq_key], []).append(self.seq_structure[seq_structure_keys[seq_key]])
            
            if seq_structure_keys[seq_key] in filter_seq_sizes:
                filter_seq_sizes.setdefault([seq_structure_keys[seq_key]], []).append(self.seq_sizes[seq_structure_keys[seq_key]])
            else:    
                filter_seq_sizes[seq_structure_keys[seq_key]] = []
                filter_seq_sizes.setdefault(seq_structure_keys[seq_key], []).append(self.seq_sizes[seq_structure_keys[seq_key]])
            
             
            #Iterate over frames in Sequence
            for frame_key in range(len(frames_of_seq)):
                
                
                #Get Filename of each Frame
                frame_name = frames_of_seq[frame_key]
                
                #Path of each Frame of this Sequence
                final_path = "{}{}{}".format(path_dir, os.sep, frame_name)
                
                #load Image
                img = cv2.imread(final_path,1)
                
                #restore RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                #resize to 256x256
                img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                #normalize
                img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                
                image_preprocessed[frame_key] = img_256_norm
                image[frame_key] = img
            
            sequence[seq_key] = image
            #expand Dimension
            sequence_preprocessed[seq_key] = np.expand_dims(image_preprocessed, 0)
            
        output = {}
        output['proportion_train_test'] = self.proportion_train_test
        output['images'] = sequence
        output['sequences_preprocessed'] = sequence_preprocessed
        output['seq_structure'] = filter_seq_structure
        output['img_sizes'] = filter_seq_sizes
        
        return output
    
    def get_one_seqence_from_mode(self, key, mode):
        """
        key = index of sequence in 'train'-part or 'test'-part of the Dataset (depends on value in mode)
        mode = 'train' or 'test'
        """
        assert mode == 'train' or mode == 'test', \
            "Unknown mode"
        
        sequence = {}
        sequence_preprocessed = {}
        #Names of sequences (0001, 0002, ...)
        seq_structure_keys = list(self.proportion_train_test[mode])
        
        assert len(seq_structure_keys) > key, \
            "Key %d ist greater than listlength %d. (Key counts from 0 to listlength-1)" % (key, len(seq_structure_keys))
            
        seq_key = key 
        
        image = {}
        image_preprocessed = {}
        #Path of each sequence
        path_dir = os.path.join(os.getcwd(), self.dataset_path)
        path_dir = "{}{}{}".format(path_dir, os.sep, 'frames')
        path_dir = "{}{}{}".format(path_dir, os.sep, seq_structure_keys[seq_key])
            
        #Frames of an indiviudal sequence
        frames_of_seq = self.seq_structure[seq_structure_keys[seq_key]]
            
        #Iterate over frames in sequence
        for frame_key in range(len(frames_of_seq)):
                
                
            #Get filename of each frame
            frame_name = frames_of_seq[frame_key]
                
            #Path of each frame of this sequence
            final_path = "{}{}{}".format(path_dir, os.sep, frame_name)
                
            #load image
            img = cv2.imread(final_path,1)
                
            #restore RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            #resize to 256x256
            img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            #normalize
            img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                
            image_preprocessed[frame_key] = img_256_norm
            image[frame_key] = img
            
        sequence[seq_key] = image
        #expand dimension
        sequence_preprocessed[seq_key] = np.expand_dims(image_preprocessed, 0)
            
        output = {}
        output['proportion_train_test'] = self.proportion_train_test
        output['images'] = sequence
        output['sequences_preprocessed'] = sequence_preprocessed
        output['seq_structure'] = self.seq_structure[seq_structure_keys[seq_key]]
        output['img_sizes'] = self.seq_sizes[seq_structure_keys[seq_key]]
        
        return output
    
    
    def get_length(self):
        
        return len(self.seq_structure)
        
    def get_proportion_train_test(self):
        
        return self.proportion_train_test
    
    def iterate_over_frames_in_sequence(self, frames_of_seq, path_dir):
        image_preprocessed = {}
        image = {}
        #Iterate over frames in Sequence
        for frame_key in range(len(frames_of_seq)):
                
                
            #Get Filename of each Frame
            frame_name = frames_of_seq[frame_key]
                
            #Path of each Frame of this Sequence
            final_path = "{}{}{}".format(path_dir, os.sep, frame_name)
                
            #load Image
            img = cv2.imread(final_path,1)
                
            #restore RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            #resize to 256x256
            img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            #normalize
            img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
                
            image_preprocessed[frame_key] = img_256_norm
            image[frame_key] = img
                
        return [image_preprocessed, image]