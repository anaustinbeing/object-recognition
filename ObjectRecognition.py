# !/usr/bin/env python

# Importing modules
import numpy as np
import cv2

# Local modules
from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: Object Recognition
Place the object to be recognized before the webcam.
Following objects are recognized:
1. Face
2. Eye
3. Smile
4. Wall Clock
5. Number Plate
'''

# Function for detecting face
def detect_face_eye(img, cascade):
    
    rects = cascade.detectMultiScale(img, scaleFactor=1.25, minNeighbors=4, \
                            minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


# Function for detecting smile
def detect_smile(img, cascade):
    
    rects = cascade.detectMultiScale(img, scaleFactor=1.23, minNeighbors=60, \
                            minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

# Function for detecting number plate
def detect_numberplate(img, cascade):
    
    rects = cascade.detectMultiScale(img, scaleFactor=1.01, minNeighbors=15, \
                            minSize=(3, 3), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

# Function for detecting wall clock
def detect_wc(img, cascade):
    
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=10, \
                            minSize=(5, 5), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

# Function to draw rectangle around the detected object
def draw_rects(img, rects, color, obj):
    
    for x1, y1, x2, y2 in rects:
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        draw_str(img, (x1, y1-10), ' # %s' %obj) # Name of the detected object
        
# Main function      
if __name__ == '__main__':
    
    import sys, getopt
    print help_message
      
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    # Invoking cascade files
    cascade_fn = args.get('--cascade', "E:\SOFTWARES\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "E:\SOFTWARES\opencv\sources\data\haarcascades\haarcascade_eye.xml")
    smile_fn = args.get('--cascade', "E:\SOFTWARES\opencv\sources\data\haarcascades\haarcascade_mcs_mouth.xml")
    numberplate_fn = args.get('--cascade', "E:\SOFTWARES\opencv\sources\data\haarcascades\haarcascade_russian_plate_number.xml")
    wc_fn = args.get('--cascade', "E:\SOFTWARES\opencv\sources\data\haarcascades\wc.xml")

    # Initializing the cascade files
    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    smile = cv2.CascadeClassifier(smile_fn)
    numberplate = cv2.CascadeClassifier(numberplate_fn)
    wc = cv2.CascadeClassifier(wc_fn)

    cam = create_capture(video_src, fallback='synth:bg=../data/aloeR.jpg:noise=0.0')
    
    while (cam.isOpened()): 
        
        ret, img = cam.read()                                           # Reading from the webcam
        
        if ret:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # Convert to grayscale
            gray = cv2.equalizeHist(gray)                               # Histogram equalization (Adjusting brightness)
            rects = detect_face_eye(gray, cascade)                      
            rect_smile = detect_smile(gray, smile)
            rect_wc = detect_wc(gray, wc)
            rect_numberplate = detect_numberplate(gray, numberplate)
            
            vis = img.copy()                                            # Copy of the original image
            
            draw_rects(vis, rects, (255, 0, 255), "Face")
            draw_rects(vis, rect_smile, (255, 255, 0), "smile")
            draw_rects(vis, rect_wc, (255, 255, 0), "WallClock")
            draw_rects(vis, rect_numberplate, (255, 255, 255), "NumberPlate")
            
            if not nested.empty():
                
                for x1, y1, x2, y2 in rects:
                    
                    roi = gray[y1:y2, x1:x2]
                    vis_roi = vis[y1:y2, x1:x2]
                    subrects = detect_face_eye(roi.copy(), nested)
                    draw_rects(vis_roi, subrects, (0, 255, 0), "Eye")
            
            
            cv2.imshow('CONTROLLED OBJECT RECOGNITION', vis)            # Displaying the window 'CONTROLLED OBJECT RECOGNITION'

            if 0xFF & cv2.waitKey(5) == 27:
                
                cam.release()                                           
                break
            
    cam.release()                                                       # Closing the webcam
    cv2.destroyAllWindows()                                             # Destroying the window
