import os
import random
import pickle
import numpy.random as npr
import copy
import cv2
import numpy as np
import time

class Benset(object):
    def __init__(self, dataset_path, num_action_predictions, use_backgrounds=False, train_data_keys = None, val_data_keys = None):
        
        self.num_action_predictions = num_action_predictions
        self.dataset_path = dataset_path
        self.use_backgrounds = use_backgrounds
        #700 Action 0
        #700 Action 1
        #700 Action 2
        #800 Action 3
    
        #(700 Clips x 2 Cams x 3) + (800 Clips x 2 Cams) = 5800 Clips
    
        #100 Clips = 1 Recording
        
        num_sequences = [700, 700, 700, 800]
        num_clips_per_recording = 100
        num_cams = 2
        
        self.dataset_annotations = {}
        self.generate_dataset_structure(num_sequences, num_cams, num_clips_per_recording, num_action_predictions)
        
        if train_data_keys is not None:
            self.train_data_keys = train_data_keys
            
            self.test_annotations = copy.deepcopy(self.dataset_annotations)
            
            for sequence in train_data_keys:
                self.test_annotations.pop(sequence)
            
            
        if val_data_keys is not None:
            self.test_data_keys = val_data_keys
            
            self.train_annotations = copy.deepcopy(self.dataset_annotations)
            
            for sequence in val_data_keys:
                self.train_annotations.pop(sequence)
                
                
            #TODO !!!!!!!!!
            self.val_data_keys = self.test_data_keys
            self.val_annotations = self.test_annotations
            
    def generate_dataset_structure(self, num_sequences, num_cams, num_clips_per_recording, num_action_predictions):
        
        if num_cams == 2:
            toggle_cam = 0
        else:
            #TODO
            toggle_cam = 0
        
        #proportion_train_test = 5 # Verhältnis 20 / 80 : Test / Train -> (1/5)
        
        self.dataset_structure = {}
        self.dataset_keys = {}
        seq_structure = []
        counter = 0
        
        for root, dirs, files in os.walk(os.path.join(os.path.join(os.getcwd(), self.dataset_path),"frames")):
           
            #Get names of sequences
            if dirs != []:
                if len(dirs) > 1:
                    seq_structure = dirs
                    
                
            #Get names of frames and picture sizes
            if files != []:
                if len(files) > 1:
                    
                    #Mapping of seqences and corresponding frames
                    self.dataset_structure[seq_structure[counter]] = files
                    counter += 1
                    
        if self.use_backgrounds:
            self.backgrounds = []          
            for root, dirs, files in os.walk(os.path.join(os.path.join(os.getcwd(), self.dataset_path),"backgrounds")):
               
                #Get names of sequences
                if dirs != []:
                    if len(dirs) > 1:
                        seq_structure = dirs
                        
                    
                #Get names of frames and picture sizes
                if files != []:
                    if len(files) > 1:
                        
                        #Mapping of seqences and corresponding frames
                        self.backgrounds = files
                        counter += 1
        
        #Split into test/val
        test_data = []
        train_data = []
        
        dataset_structure_keys = list(self.dataset_structure.keys())
        self.dataset_keys = copy.deepcopy(dataset_structure_keys)
        split_train_val_test = copy.deepcopy(dataset_structure_keys)
        
        random.seed(int(time.time()))
        self.train_data_keys = np.random.choice(dataset_structure_keys, size=int((len(dataset_structure_keys) * 0.8)), replace=False)
        
        for sequence in self.train_data_keys:
            split_train_val_test.remove(sequence)
        
        random.seed(int(time.time())+1)
        self.test_data_keys = np.random.choice(split_train_val_test, size=int((len(split_train_val_test) * 0.5)), replace=False)
        
        for sequence in self.test_data_keys:
            split_train_val_test.remove(sequence)
        
        self.val_data_keys = np.asarray(split_train_val_test)
                 
        #counter = 1
        #for sequence in self.dataset_keys:
        #    if proportion_train_test == counter:
        #        test_data.append(sequence)
        #        counter = 1
        #    else:
        #        train_data.append(sequence)
        #        counter += 1          
        
        #self.train_data_keys = train_data
        #random.seed(int(time.time()))
        #random.shuffle(self.train_data_keys)
        #self.test_data_keys = test_data
        #random.seed(int(time.time()))
        #random.shuffle(self.test_data_keys)
        #self.pool_of_train_data = copy.deepcopy(self.train_data_keys)
        
        for sequence in dataset_structure_keys:
            actions = []
            if sequence.find("A00000") == 12:
                for i in range(num_action_predictions):
                    actions.append(np.reshape(np.array([1,0,0,0]), (1,4)))
                self.dataset_annotations.update({sequence:actions})
            if sequence.find("A00001") == 12:
                for i in range(num_action_predictions):
                    actions.append(np.reshape(np.array([0,1,0,0]), (1,4)))
                self.dataset_annotations.update({sequence:actions})
            if sequence.find("A00002") == 12:
                for i in range(num_action_predictions):
                    actions.append(np.reshape(np.array([0,0,1,0]), (1,4)))
                self.dataset_annotations.update({sequence:actions})
            if sequence.find("A00003") == 12:
                for i in range(num_action_predictions):
                    actions.append(np.reshape(np.array([0,0,0,1]), (1,4)))
                self.dataset_annotations.update({sequence:actions})
                
        self.train_annotations = copy.deepcopy(self.dataset_annotations)
        self.test_annotations = copy.deepcopy(self.dataset_annotations)
        self.val_annotations = copy.deepcopy(self.dataset_annotations)
        
        for sequence in self.train_data_keys:
            self.test_annotations.pop(sequence)
            self.val_annotations.pop(sequence)
    
        for sequence in self.test_data_keys:
            self.train_annotations.pop(sequence)
            self.val_annotations.pop(sequence)
            
        for sequence in self.val_data_keys:
            self.train_annotations.pop(sequence)
            self.test_annotations.pop(sequence)
            
            
        #self.load_pictures()
    
    def get_dataset_length(self):
        
        return len(self.test_data_keys) + len(self.train_data_keys)
    
    def get_train_data_length(self):
        
        return len(self.train_data_keys)
    
    def get_test_data_length(self):
        
        return len(self.test_data_keys)
        
    def get_dataset_keys(self):
        
        return self.dataset_keys
    
    def get_train_data_keys(self):
        
        return self.train_data_keys
    
    def get_test_data_keys(self):
        
        return self.test_data_keys
    
    def get_val_data_keys(self):
        
        return self.val_data_keys
    
    def get_dataset_structure(self):
        
        return self.dataset_structure
    
    def get_dataset_path(self):
        
        return self.dataset_path
    
    def get_dataset_annotations(self):
        
        return self.dataset_annotations
    
    def get_train_annotations(self):
        
        return self.train_annotations
    
    def get_val_annotations(self):
        
        return self.val_annotations
    
    def get_test_annotations(self):
        
        return self.test_annotations
    
    def get_backgrounds(self):
        
        return self.backgrounds
    
    def load_pictures(self):
        
        self.dataset = {}
        
        for sequence in self.dataset_structure:
            #Path to Clip
            path = self.get_dataset_path()
            path += "\\frames\\"
            path += sequence
            
            seq = []
            
            for frame in self.dataset_structure[sequence]:
            
                #Path of each Frame of this Sequence
                final_path = "{}{}{}".format(path, os.sep, frame)
            
                img = cv2.imread(final_path,1)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                seq.append(img)
                
            self.dataset.update({sequence:seq})
                
        
    def get_num_action_predictions(self):
        
        return self.num_action_predictions