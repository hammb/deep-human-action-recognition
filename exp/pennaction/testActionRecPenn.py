import glob
import cv2

#Read cropped Images
images = [cv2.imread(file) for file in glob.glob("datasets/Squad/*.png")]

img_256_norm = []
for videoIndex in range(len(images)):
    #resize to 256x256
    img_256 = cv2.resize(images[videoIndex], (256,256), interpolation = cv2.INTER_AREA)
    #normalize
    img_256_norm.append(cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F))
    
