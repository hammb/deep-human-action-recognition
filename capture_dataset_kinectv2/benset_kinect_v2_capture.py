import os
import sys
import time
import numpy as np
from PIL import Image
import cv2
import winsound
import threading

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


def capture(sequence,num_seq, duration_of_time_between_capture, duration, action, cam):

    assert num_seq > 0, \
        "Number of Sequneces has to be > 0"
    
    assert duration_of_time_between_capture > 0, \
        "Duration of pause has to be > 0"
        
    assert duration > 0, \
        "Duration of capture has to be > 0"
        
    #action: Action 0: Hemd an, Action 1: Hemd aus, Action 2: Sprize auffüllen, Action 3: nichts     
    assert action in [0,1,2,3], \
        "Invalid action: %d" % action
        
    #cam: Kinect Cam No. 1 or 2
    assert cam in [0,1], \
        "Invalid cam: %d" % cam

    frequency = 700  # Set Frequency To 2500 Hertz
    
    helper = 1 #create directory at first run
    
    
    start_time = 0
    frame_counter = 0
    winsound.Beep(frequency, duration_of_time_between_capture)
    while num_seq > sequence: 
        
        if kinect.has_new_color_frame():
            
            #create directory at first run
            if helper:
                
                start_time = time.time()
                
                directory = "S%05dC%05dA%05d" % (sequence, cam, action)
                
                try:
                    os.mkdir("E:\\Bachelorarbeit-SS20\\datasets\\Benset256_testdataset\\frames\\" + directory)
                except:
                    print("Directory '" + directory + "' already exists in this path")
                
                
                helper = 0
            
            frame = kinect.get_last_color_frame()
            
            colourframe = np.reshape(frame, (2073600, 4))
            colourframe = colourframe[:,0:3]
            
            #extract then combine the RBG data
            colourframeR = colourframe[:,0]
            colourframeR = np.reshape(colourframeR, (1080, 1920))
            colourframeG = colourframe[:,1]
            colourframeG = np.reshape(colourframeG, (1080, 1920))        
            colourframeB = colourframe[:,2]
            colourframeB = np.reshape(colourframeB, (1080, 1920))
            
            framefullcolour = cv2.merge([colourframeR, colourframeG, colourframeB])
            framefullcolour = cv2.cvtColor(framefullcolour, cv2.COLOR_BGR2RGB)
            
            img = cv2.cvtColor(framefullcolour, cv2.COLOR_BGR2RGB)
            
            x = int((len(img[0]) - len(img)) / 2)
            w = int(len(img))
            y = 0
            h = int(len(img))
            
            img = img[y:y+h, x:x+w]
            
            img_256 = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
        
            img_256 = cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB)
        
            cv2.imwrite("E:\\Bachelorarbeit-SS20\\datasets\\Benset256_testdataset\\frames\\" + directory + "\\%05d.jpg" % (frame_counter), img) 
            
            frame_counter = frame_counter + 1
            frame = None
            
            
        
        if int((start_time + (duration/1000))) == int(time.time()):
            print (sequence)
            helper = 1 #create directory at first run
            start_time = 0
            frame_counter = 0
            sequence += 1
            
            winsound.Beep(frequency, duration_of_time_between_capture)
            
    
        
    kinect.close()
    
"""Load kinect"""

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
#Starts at next full minute
while True:
    if (int(time.time()) % 60 == 0):
        capture(0, 50, 10000, 5000, 2, 0)
        winsound.Beep(1000, 1000)
        break
