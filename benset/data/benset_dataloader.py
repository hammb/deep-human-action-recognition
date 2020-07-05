import os
import random
import pickle
import numpy.random as npr

class Benset(object):
    def __init__(self, dataset_path, pose_predictons_path, dataset_structure_file_path=None, adjust_dataset_structure=False, test_data_file_path=None):
        
        self.test_annotations = {}
        self.train_annotations = {}
        
        self.dataset_path = dataset_path
        
        with open(pose_predictons_path, 'rb') as fp:
            self.pose_predictons = pickle.load(fp)
        
        if dataset_structure_file_path is None:  
            
            #700 Action 0
            #700 Action 1
            #700 Action 2
            #800 Action 3
        
            #(700 Clips x 2 Cams x 3) + (800 Clips x 2 Cams) = 5800 Clips
        
            #100 Clips = 1 Recording
            
            num_sequences = [700, 700, 700, 800]
            num_clips_per_recording = 100
            num_cams = 2
            
            #TODO ANNOTATIONEN
            
            #self.dataset_annotations = self.pose_predictons
            #self.train_annotations = self.pose_predictons
            
            self.generate_dataset_structure(num_sequences, num_cams, num_clips_per_recording)
            
        else:
            with open(dataset_structure_file_path, 'rb') as fp:
                annotation_file = pickle.load(fp)
            
            #Kürzt alle Klassen auf die Länge der kürzesten
            if adjust_dataset_structure:
                self.adjust_dataset_structure(annotation_file, test_data_file_path)
            else:
                self.dataset_structure = annotation_file
                #TODO ANNOTATIONEN
        
    def generate_dataset_structure(self, num_sequences, num_cams, num_clips_per_recording):
        
        if num_cams == 2:
            toggle_cam = 0
        else:
            #TODO
            toggle_cam = 0
        
        proportion_train_test = 5 # Verhältnis 20 / 80 : Test / Train -> (1/5)
        
        self.dataset_structure = {}
        self.dataset = {}
        counter = 0
        
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), self.dataset_path)):
           
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
        
        #Split into test/val
        test_data = []
        train_data = []
        
        dataset_structure_keys = list(self.dataset_structure.keys())
        self.dataset = dataset_structure_keys
        
        for a in range(len(num_sequences)): #0-4 Iterate over Actions
            for s in range(num_sequences[a] // 100): #0-7 (0-8)
                counter = 1
                for clip in range(num_clips_per_recording): #0-100
                    
                    if proportion_train_test == counter:
                        
                        test_data.append("S%05dC%05dA%05d" % (s*clip, toggle_cam, a))
                        toggle_cam ^= 1
                        test_data.append("S%05dC%05dA%05d" % (s*clip, toggle_cam, a))
                        
                        counter = 1
                    else:
                        
                        train_data.append("S%05dC%05dA%05d" % (s*clip, toggle_cam, a))
                        toggle_cam ^= 1
                        train_data.append("S%05dC%05dA%05d" % (s*clip, toggle_cam, a))
                        
                        counter += 1
                    
                    
        self.train_data = train_data
        self.test_data = test_data
        
        #for sequence in test_data:
        #    self.train_annotations.pop(sequence)
            
    
    def get_dataset_length(self):
        
        return len(self.test_data) + len(self.train_data)
    
    def get_train_data_length(self):
        
        return len(self.train_data)
    
    def get_test_data_length(self):
        
        return len(self.test_data)
        
    def get_dataset(self):
        
        return self.dataset
    
    def get_train_data(self):
        
        return self.train_data
    
    def get_test_data(self):
        
        return self.test_data
    
    def get_dataset_structure(self):
        
        return self.dataset_structure
    
    def get_dataset_path(self):
        
        return self.dataset_path
    
    def get_dataset_annotations(self):
        
        return self.dataset_annotations
    
    def get_train_annotations(self):
        
        return self.train_annotations
    
    def get_test_annotations(self):
        
        return self.test_annotations
    
    def adjust_dataset_structure(self, dataset_structure, test_data_file_path=None):
        
        
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
        
        all_seqences_unabridged = countA0 + countA1 + countA2 + countA3 #ungekürzt alle Sequenzen
        
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
            
        self.dataset_structure = dataset_structure
        
        dataset_structure_keys = list(self.dataset_structure.keys())
        self.dataset = dataset_structure_keys
        
        #Split into test/val
        proportion_train_test = 5 # Verhältnis 20 / 80 : Test / Train -> (1/5)
        
        all_seqences = countA0 + countA1 + countA2 + countA3
        
        if test_data_file_path is None:
            self.test_data = npr.choice(all_seqences, size=int((len(all_seqences)/proportion_train_test)), replace=False)
            
            with open("output\\test_data.p", 'wb') as fp:
                pickle.dump(self.test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
        else:
            with open(test_data_file_path, 'rb') as fp:
                test_data_file = pickle.load(fp)
            
            self.test_data = test_data_file
        
        #self.test_annotations = self.pose_predictons
        #self.train_annotations = self.pose_predictons
        
        for sequences in self.test_data:
            self.test_annotations.update({sequences:self.pose_predictons[]})
        
        for test_sequence in self.test_data:
            
            a = all_seqences_unabridged.index(test_sequence)
            b = all_seqences.index(test_sequence)
            
            all_seqences_unabridged.pop(a)
            all_seqences.pop(b)
            
        self.pool_of_train_data = all_seqences_unabridged
        self.train_data = all_seqences
        
        
    def shuffle_train_data(self):
        self.train_data = random.choices(self.pool_of_train_data, k=len(self.train_data))
        random.shuffle(self.train_data)