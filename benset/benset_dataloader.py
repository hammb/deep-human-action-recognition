import os

class Benset(object):
    def __init__(self, dataset_path, annotation_file=None):
        self.dataset_path = dataset_path
        self.annotation_file = annotation_file
        
        if self.annotation_file is None:  
            self.generate_annotations()
        
    def generate_annotations(self):
        
        proportion_train_test = 4
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
                    
                    #Division into test and training data
                    if counter%proportion_train_test == 0:
                        if 'test' in self.dataset:
                            self.dataset.setdefault('test', []).append(root)
                        else: 
                            self.dataset['test'] = []
                            self.dataset.setdefault('test', []).append(root)
                    else:
                        if 'train' in self.dataset:
                            self.dataset.setdefault('train', []).append(root)
                        else: 
                            self.dataset['train'] = []
                            self.dataset.setdefault('train', []).append(root)
                    
                    #Mapping of seqences and corresponding frames
                    self.dataset_structure[seq_structure[counter]] = files
                    counter += 1
                    
    
    def get_dataset_length(self):
        
        return len(self.dataset['test']) + len(self.dataset['train'])
    
    def get_train_data_length(self):
        
        return len(self.dataset['train'])
    
    def get_test_data_length(self):
        
        return len(self.dataset['test'])
        
    def get_dataset(self):
        
        return self.dataset
    
    def get_train_data(self):
        
        return self.dataset['train']
    
    def get_test_data(self):
        
        return self.dataset['test']
    
    def get_dataset_structure(self):
        
        return self.dataset_structure
    