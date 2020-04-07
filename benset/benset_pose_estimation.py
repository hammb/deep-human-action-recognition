import os
import sys
        


sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_dataloader import Benset
benset = Benset('datasets/Benset')

a = benset.get_dataset()

b = benset.get_one_seqence_from_dataset(0)

c = benset.get_data_from_mode('train')

d = benset.get_one_seqence_from_mode(0, 'train')