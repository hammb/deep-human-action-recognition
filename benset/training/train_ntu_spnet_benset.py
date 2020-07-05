import os
import sys
import pickle
import cv2
import numpy as np
#if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
#    sys.path.append(os.getcwd())
from collections import Counter
from sklearn.metrics import confusion_matrix
import deephar

from deephar.config import mpii_dataconf
from deephar.config import human36m_dataconf
from deephar.config import ntu_dataconf
from deephar.config import ntu_pe_dataconf
from deephar.config import ModelConfig
from deephar.config import DataConfig

from deephar.data import MpiiSinglePerson
from deephar.data import Human36M
from deephar.data import PennAction
from deephar.data import Ntu
from deephar.data import BatchLoader

from deephar.losses import pose_regression_loss
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import keras.backend as K

from deephar.callbacks import SaveModel

from deephar.trainer import MultiModelTrainer
from deephar.models import compile_split_models
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
#from datasetpath import datasetpath

from mpii_tools import MpiiEvalCallback
from h36m_tools import H36MEvalCallback
from ntu_tools import NtuEvalCallback

logdir = 'E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0625\\logs'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
cfg = ModelConfig((num_frames,) + ntu_dataconf.input_shape, pa17j3d,
        # num_actions=[60], num_pyramids=8, action_pyramids=[5, 6, 7, 8],
        num_actions=[4], num_pyramids=2, action_pyramids=[1, 2],
        num_levels=4, pose_replica=False,
        num_pose_features=192, num_visual_features=192)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

start_lr = 0.01
action_weight = 0.1
#batch_size_mpii = 3
#batch_size_h36m = 4
#batch_size_ntu = 6 #1
#batch_clips = 1 # 8/4


"""Build the full model"""
full_model = spnet.build(cfg)

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(
         "E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\ntu_baseline\\0603\\weights_posebaseline_060.hdf5",
         by_name=True)

"""Save model callback."""

save_model = SaveModel(os.path.join("E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0625",
    'weights_3dp+benset_ar_{epoch:03d}.hdf5'), model_to_save=full_model)




#with open("C:\\networks\\deephar\\output\\split_train_val_benset\\train_list.p", 'rb') as fp:
#    train_data_keys = pickle.load(fp)

#with open("C:\\networks\\deephar\\output\\split_train_val_benset\\val_list.p", 'rb') as fp:
#    val_data_keys = pickle.load(fp)

sys.path.append(os.path.join(os.getcwd(), 'benset'))   
     
from benset_batchloader_benset import *
from benset_dataloader_ar import *


dataset_path_green="E:\\Bachelorarbeit-SS20\\datasets\\Benset256_green"

num_action_predictions = 6
use_backgrounds = True

benset_dataloader = Benset(dataset_path_green, num_action_predictions,
                      use_backgrounds=use_backgrounds)

batch_size = 3
num_frames = 8
mode=1
green_screen = 1
augmentation = 1

benset_train_batchloader = BatchLoader(dataloader=benset_dataloader, x_set=benset_dataloader.get_train_data_keys(),
                         y_set=benset_dataloader.get_train_annotations(),batch_size=batch_size,
                         num_frames=num_frames,mode=mode,green_screen=green_screen,backgrounds=benset_dataloader.get_backgrounds(),
                         augmentation=augmentation)

batch_size = 1
augmentation = 0
benset_val_batchloader = BatchLoader(dataloader=benset_dataloader, x_set=benset_dataloader.get_val_data_keys(),
                         y_set=benset_dataloader.get_val_annotations(),batch_size=batch_size,
                         num_frames=num_frames,mode=mode, augmentation=augmentation)


benset_test_batchloader = BatchLoader(dataloader=benset_dataloader, x_set=benset_dataloader.get_test_data_keys(),
                         y_set=benset_dataloader.get_test_annotations(),batch_size=batch_size,
                         num_frames=num_frames,mode=mode, augmentation=augmentation)


logarray = {}

def prepare_training(pose_trainable, lr):
    optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
    # optimizer = RMSprop(lr=lr)
    models = compile_split_models(full_model, cfg, optimizer,
            pose_trainable=pose_trainable, ar_loss_weights=action_weight,
            copy_replica=cfg.pose_replica)
    full_model.summary()

    """Create validation callbacks."""
    mpii_callback = MpiiEvalCallback(mpii_x_val, mpii_p_val, mpii_afmat_val,
            mpii_head_val, eval_model=models[0], pred_per_block=1,
            map_to_pa16j=pa17j3d.map_to_pa16j, batch_size=1, logdir=logdir)

    h36m_callback = H36MEvalCallback(h36m_x_val, h36m_pw_val, h36m_afmat_val,
            h36m_puvd_val[:,0,2], h36m_scam_val, h36m_action,
            batch_size=1, eval_model=models[0], logdir=logdir)

    ntu_callback = NtuEvalCallback(ntu_te, eval_model=models[1], logdir=logdir)

    def end_of_epoch_callback(epoch):
        
        save_model.on_epoch_end(epoch)
        
        y_actu = []
        y_pred = []
        predictions = []
        printcn(OKBLUE, 'Validation on Benset')
        for i in range(len(benset_dataloader.get_val_data_keys())):
            #printc(OKBLUE, '%04d/%04d\t' % (i, len(val_data_keys)))
            
            x , y = benset_val_batchloader.__next__()
            prediction = full_model.predict(x)
            
            pred_action = np.argmax(prediction[11])
            annot_action = np.argmax(y[0])
            
            y_actu.append(annot_action)
            y_pred.append(pred_action)
            
            if pred_action == annot_action:
                predictions.append(1)
            else:
                predictions.append(0)
                
            accuracy = 100.0 / len(predictions) * Counter(predictions)[1]
            
        conf_mat = confusion_matrix(y_actu, y_pred)
        printcn(OKBLUE, '')
        printcn(OKBLUE, 'Accuracy: %d' % accuracy)
        print(conf_mat)
        
        logarray[epoch] = accuracy
            
        with open(os.path.join(logdir, 'benset_val.json'), 'w') as f:
                json.dump(logarray, f)
            
        # if epoch == 0 or epoch >= 50:
        # mpii_callback.on_epoch_end(epoch)
        # h36m_callback.on_epoch_end(epoch)

        #ntu_callback.on_epoch_end(epoch)

        if epoch in [25, 31]:
            lr = float(K.get_value(optimizer.lr))
            newlr = 0.1*lr
            K.set_value(optimizer.lr, newlr)
            printcn(WARNING, 'lr_scheduler: lr %g -> %g @ %d' \
                    % (lr, newlr, epoch))

    return end_of_epoch_callback, models


#95153
steps_per_epoch = benset_train_batchloader.__len__()

fcallback, models = prepare_training(False, start_lr)
# trainer = MultiModelTrainer(models[1:], [ar_data_tr], workers=8,
        # print_full_losses=True)
# trainer.train(50, steps_per_epoch=steps_per_epoch, initial_epoch=0,
        # end_of_epoch_callback=fcallback)

"""Joint learning the full model."""
# steps_per_epoch = mpii.get_length(TRAIN_MODE) // (batch_size_mpii * batch_clips)

fcallback, models = prepare_training(True, 0.1*start_lr)
# trainer = MultiModelTrainer(models, [pe_data_tr, ar_data_tr], workers=8,
trainer = MultiModelTrainer([models[1]], [benset_train_batchloader], workers=6,
        print_full_losses=True)
trainer.train(40, steps_per_epoch=steps_per_epoch, initial_epoch=3,
        end_of_epoch_callback=fcallback)

y_actu = []
y_pred = []
predictions = []
printcn(OKBLUE, 'Validation on Benset')
for i in range(len(benset_dataloader.get_test_data_keys())):
    #printc(OKBLUE, '%04d/%04d\t' % (i, len(val_data_keys)))
    
    x , y = benset_test_batchloader.__next__()
    prediction = full_model.predict(x)
    
    pred_action = np.argmax(prediction[11])
    annot_action = np.argmax(y[0])
    
    y_actu.append(annot_action)
    y_pred.append(pred_action)
    
    if pred_action == annot_action:
        predictions.append(1)
    else:
        predictions.append(0)
        
    accuracy = 100.0 / len(predictions) * Counter(predictions)[1]
    
conf_mat = confusion_matrix(y_actu, y_pred)
printcn(OKBLUE, '')
printcn(OKBLUE, 'Accuracy: %d' % accuracy)
print(conf_mat)


"""

EVALUATE ON TESTDATA

"""

full_model.load_weights(
         "E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0623\\weights_3dp+benset_ar_026.hdf5",
         by_name=True)

dataset_structure = {}
counter = 0
dataset_path_end = "E:\\Bachelorarbeit-SS20\\datasets\\Benset256_testdataset"

for root, dirs, files in os.walk(os.path.join(os.getcwd(), dataset_path_end)):
    
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


y_actu = []
y_pred = []
predictions = []
pred_for_whole_seqs = []
counter_list = 1
for sequence in dataset_structure:
    
    
    pred_for_one_seq = []
    
    sequence_lenght = len(dataset_structure[sequence]) // num_frames
    printcn(OKBLUE, '%03d/%03d\t' % (counter_list, len(dataset_structure)))
    for frames in range(sequence_lenght):
        
        batch = dataset_structure[sequence][(frames)*num_frames:(frames+1)*num_frames]
        
        
        batch_x = []
        for frame in batch:
            img = cv2.imread(os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), dataset_path_end),"frames"),sequence),frame),1)
            img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            #img_256 = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)
            batch_x.append(img_256)
            
        batch_x = np.expand_dims(batch_x, axis = 0)
        prediction = full_model.predict(batch_x)
        
        pred_action = np.argmax(prediction[11])
        
        if sequence.find("A00000") == 12:
            annot_action = 0
        if sequence.find("A00001") == 12:
            annot_action = 1
        if sequence.find("A00002") == 12:
            annot_action = 2
        if sequence.find("A00003") == 12:
            annot_action = 3
        
        if pred_action == annot_action:
            pred_for_one_seq.append(1)
            predictions.append(1)
        else:
            pred_for_one_seq.append(0)
            predictions.append(0)
            
        accuracy = 100.0 / len(predictions) * Counter(predictions)[1]
    if Counter(pred_for_one_seq)[1] > Counter(pred_for_one_seq)[0]:
        pred_for_whole_seqs.append(1)
    else:
        pred_for_whole_seqs.append(0)
        
        
    accuracy_w = 100.0 / len(pred_for_whole_seqs) * Counter(pred_for_whole_seqs)[1]
    
    printcn(OKBLUE,'%d' % accuracy)
    printcn(OKBLUE,'%d' % accuracy_w)
    
    counter_list = counter_list + 1
conf_mat = confusion_matrix(y_actu, y_pred)
print(conf_mat)

"""
EVAL WITH BATCHLOADER
"""

full_model.load_weights(
         "E:\\Bachelorarbeit-SS20\\weights\\deephar\\output\\spnet\\0625\\weights_3dp+benset_ar_004.hdf5",
         by_name=True)

dataset_path_test="E:\\Bachelorarbeit-SS20\\datasets\\Benset256_testdataset"

num_action_predictions = 6
use_backgrounds = True

benset_dataloader_test = Benset(dataset_path_test, num_action_predictions,
                      use_backgrounds=use_backgrounds)

batch_size = 1
num_frames = 8
mode=1
green_screen = 0
augmentation = 0

benset_test_batchloader_test = BatchLoader(dataloader=benset_dataloader_test, x_set=benset_dataloader_test.get_dataset_keys(),
                         y_set=benset_dataloader_test.get_dataset_annotations(),batch_size=batch_size,
                         num_frames=num_frames,mode=mode, augmentation=augmentation)


    
y_actu = []
y_pred = []
predictions = []
printcn(OKBLUE, 'Validation on Benset')
for i in range(len(benset_dataloader_test.get_dataset_keys())):
    #printc(OKBLUE, '%04d/%04d\t' % (i, len(val_data_keys)))
    
    x , y = benset_test_batchloader_test.__next__()
    prediction = full_model.predict(x)
    
    images_with_poses = []
    
    #for frame_idx, frame in enumerate(x[0,:]):
        
        #images_with_poses.append(draw_pose_on_image(frame,prediction[5][0][frame_idx]))
    
    pred_action = np.argmax(prediction[11])
    annot_action = np.argmax(y[0])
    
    y_actu.append(annot_action)
    y_pred.append(pred_action)
    
    if pred_action == annot_action:
        predictions.append(1)
    else:
        predictions.append(0)
        
    accuracy = 100.0 / len(predictions) * Counter(predictions)[1]
    
    
conf_mat = confusion_matrix(y_actu, y_pred)
printcn(OKBLUE, '')
printcn(OKBLUE, 'Accuracy: %d' % accuracy)
print(conf_mat)

"""
Pose
"""

def draw_pose_on_image(img, prediction):
    
    
    
    pred_x_y_1 = prediction[:,0:2]
    pred_x_y_1080 = np.interp(pred_x_y_1, (0, 1), (0, 256))
    
    joints = []
    for x in prediction[:,3]:
        if x > 0.5:
            joints.append(1)
        else:
            joints.append(0)
         
    index = 0
    for x in pred_x_y_1080:
        if prediction[:,3][index] > 0.5:  
            img_out = cv2.circle(img, (int(x[0]), int(x[1])), 4, (0, 0, 255), -1)
            
        index = index + 1
    
    
    #Wirbelsäule
    if joints[3] and joints[2]:
        x1 = int(pred_x_y_1080[3][0])
        y1 = int(pred_x_y_1080[3][1])
        
        x2 = int(pred_x_y_1080[2][0])
        y2 = int(pred_x_y_1080[2][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
    
    if joints[2] and joints[1]:
        x1 = int(pred_x_y_1080[2][0])
        y1 = int(pred_x_y_1080[2][1])
        
        x2 = int(pred_x_y_1080[1][0])
        y2 = int(pred_x_y_1080[1][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
        
    if joints[1] and joints[16]:
        x1 = int(pred_x_y_1080[1][0])
        y1 = int(pred_x_y_1080[1][1])
        
        x2 = int(pred_x_y_1080[16][0])
        y2 = int(pred_x_y_1080[16][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
        
    if joints[0] and joints[16]:
        x1 = int(pred_x_y_1080[0][0])
        y1 = int(pred_x_y_1080[0][1])
        
        x2 = int(pred_x_y_1080[16][0])
        y2 = int(pred_x_y_1080[16][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
        
    #linker Arm
    
    if joints[4] and joints[2]:
        x1 = int(pred_x_y_1080[2][0])
        y1 = int(pred_x_y_1080[2][1])
        
        x2 = int(pred_x_y_1080[4][0])
        y2 = int(pred_x_y_1080[4][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
        
    if joints[4] and joints[6]:
        x1 = int(pred_x_y_1080[6][0])
        y1 = int(pred_x_y_1080[6][1])
        
        x2 = int(pred_x_y_1080[4][0])
        y2 = int(pred_x_y_1080[4][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
    
    if joints[6] and joints[8]:
        x1 = int(pred_x_y_1080[6][0])
        y1 = int(pred_x_y_1080[6][1])
        
        x2 = int(pred_x_y_1080[8][0])
        y2 = int(pred_x_y_1080[8][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2)
    
    
    #Rechter Arm
    
    if joints[2] and joints[5]:
        x1 = int(pred_x_y_1080[2][0])
        y1 = int(pred_x_y_1080[2][1])
        
        x2 = int(pred_x_y_1080[5][0])
        y2 = int(pred_x_y_1080[5][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
    if joints[5] and joints[7]:
        x1 = int(pred_x_y_1080[7][0])
        y1 = int(pred_x_y_1080[7][1])
        
        x2 = int(pred_x_y_1080[5][0])
        y2 = int(pred_x_y_1080[5][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
    if joints[7] and joints[9]:
        x1 = int(pred_x_y_1080[7][0])
        y1 = int(pred_x_y_1080[7][1])
        
        x2 = int(pred_x_y_1080[9][0])
        y2 = int(pred_x_y_1080[9][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
        
        
    
    #Hüfte
    
    if joints[0] and joints[11]:
        x1 = int(pred_x_y_1080[0][0])
        y1 = int(pred_x_y_1080[0][1])
        
        x2 = int(pred_x_y_1080[11][0])
        y2 = int(pred_x_y_1080[11][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 2)
        
    if joints[0] and joints[10]:
        x1 = int(pred_x_y_1080[0][0])
        y1 = int(pred_x_y_1080[0][1])
        
        x2 = int(pred_x_y_1080[10][0])
        y2 = int(pred_x_y_1080[10][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 2)
        
        
    #Linkes Bein
    if joints[10] and joints[12]:
        x1 = int(pred_x_y_1080[10][0])
        y1 = int(pred_x_y_1080[10][1])
        
        x2 = int(pred_x_y_1080[12][0])
        y2 = int(pred_x_y_1080[12][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
        
    if joints[12] and joints[14]:
        x1 = int(pred_x_y_1080[12][0])
        y1 = int(pred_x_y_1080[12][1])
        
        x2 = int(pred_x_y_1080[14][0])
        y2 = int(pred_x_y_1080[14][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
        
    
        
    
    #Rechtes Bein
    if joints[13] and joints[11]:
        x1 = int(pred_x_y_1080[11][0])
        y1 = int(pred_x_y_1080[11][1])
        
        x2 = int(pred_x_y_1080[13][0])
        y2 = int(pred_x_y_1080[13][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
    
    if joints[13] and joints[15]:
        x1 = int(pred_x_y_1080[15][0])
        y1 = int(pred_x_y_1080[15][1])
        
        x2 = int(pred_x_y_1080[13][0])
        y2 = int(pred_x_y_1080[13][1])
        
        img_out = cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
        
    cv2.imshow('image',img_out)
    cv2.waitKey(0)
        
    
        
        
    return img_out
