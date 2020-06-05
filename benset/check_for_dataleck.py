import os
import sys
import time
import collections

sys.path.append(os.path.join(os.getcwd(), 'benset')) 
from benset_batchloader_ar import *
from benset_dataloader_ar import *


dataset_path="E:\\Bachelorarbeit-SS20\\datasets\\Benset256"
#dataset_path="D:\\sortout_Benset"
num_action_predictions = 6

for i in range(100):
    
    print(i)
    
    benset1 = Benset(dataset_path, num_action_predictions)
    benset2 = Benset(dataset_path, num_action_predictions)
    
    if not collections.Counter(benset1.get_train_data_keys()) == collections.Counter(benset2.get_train_data_keys()):
        print("DATALECK 1")
        
    for key in benset1.get_test_data_keys():
        if key in benset1.get_train_data_keys():
            print("DATALECK 2")
            
    for key in benset2.get_test_data_keys():
        if key in benset2.get_train_data_keys():
            print("DATALECK 3")
            
benset = Benset(dataset_path, num_action_predictions)