import os

class Benset(object):
    def __init__(self, dataset_path, annotation_file=None):
        
        #700 Action 0
        #700 Action 1
        #700 Action 2
        #800 Action 3
        
        num_sequences = [700, 700, 700, 800]
        
        num_cams = 2
        
        # (700 Clips x 2 Cams x 3) + (800 Clips x 2 Cams) = 5800 Clips
        
        # 100 Clips = 1 Recording
        
        num_clips_per_recording = 100
        
        self.dataset_path = dataset_path
        self.annotation_file = annotation_file
        
        if self.annotation_file is None:  
            self.generate_annotations(num_sequences, num_cams, num_clips_per_recording)
        
    def generate_annotations(self, num_sequences, num_cams, num_clips_per_recording):
        
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
    