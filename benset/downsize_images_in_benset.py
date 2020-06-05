import os
import cv2

dataset_structure = {}
counter = 0
dataset_path = "E:\\Bachelorarbeit-SS20\\datasets\\Benset256"

for root, dirs, files in os.walk(os.path.join(os.getcwd(), dataset_path)):
    
    #Get names of sequences
    if dirs != []:
        if len(dirs) > 1:
            seq_structure = dirs
             
         
    #Get names of frames
    if files != []:
        if len(files) > 1:
             
        #Mapping of seqences and corresponding frames
            dataset_structure[seq_structure[counter]] = files
            counter += 1

for sequence in seq_structure:
    
    try:
        os.mkdir("E:\\Bachelorarbeit-SS20\\datasets\\Benset256_green\\frames\\" + sequence)
    except:
        print("Directory '" + sequence + "' already exists in this path")
    
    for frame_index in range(len(dataset_structure[sequence])):
        
        path = dataset_path
        path += "\\frames\\"
        path += sequence
        path += "\\"
        path += dataset_structure[sequence][frame_index]
        
        img = cv2.imread(path,1)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #resize to 256x256
        img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
        
        img_256 = cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite("E:\\Bachelorarbeit-SS20\\datasets\\Benset256\\frames\\" + sequence + "\\" + dataset_structure[sequence][frame_index], img_256)
            
        