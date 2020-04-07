import os
import sys
 
sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader import *
from benset_dataloader import *

benset = Benset('datasets/Benset')

benset_seq = BatchLoader(benset.get_dataset_structure(), benset.get_test_data(),[0,0],1)