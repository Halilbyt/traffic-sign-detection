from sys import flags
import numpy as np
import cv2 as cv
#import tensorflow as tf

path = r'haarcascades\haarcascade_eye_tree_eyeglasses.xml'
path2 = r'haarcascades\haarcascade_eye.xml'
path_left  = r'haarcascades\haarcascade_lefteye_2splits.xml'
path_right = r'haarcascades\haarcascade_righteye_2splits.xml'
path_smile = r'haarcascades\haarcascade_smile.xml'
goz_obje = cv.CascadeClassifier(path_left)
goz_obje2 = cv.CascadeClassifier(path_right)
cap = cv.VideoCapture(0)

def tespit(frame):
    frame_copy = frame.copy()
    frame_copy_gray = cv.cvtColor(frame_copy,cv.COLOR_BGR2GRAY)
    frame_copy_gray = cv.equalizeHist(frame_copy_gray)
    goz_tespit1 = goz_obje.detectMultiScale(frame_copy_gray,
                                        scaleFactor = 1.05,
                                        minSize = (30,30),
                                        minNeighbors = 5,
                                        flags=cv.CASCADE_SCALE_IMAGE) 

    goz_tespit2 = goz_obje2.detectMultiScale(frame_copy_gray,
                                        scaleFactor = 1.05,
                                        minNeighbors = 5,
                                        minSize = (30,30),
                                        flags=cv.CASCADE_SCALE_IMAGE) 
# scaleFactor = 1.05, minNeighbors = 5, minSize = (30,30)
# flags=cv2.CASCADE_SCALE_IMAGE
    if len(goz_tespit1) & len(goz_tespit2) == 0:                                
        cv.putText(frame_copy,'Gozler KAPALI',(20,100),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    else:
        for (x,y,w,h) in goz_tespit1:
            cv.rectangle(frame_copy,(x,y),(x+w,y+h),(0,255,0),5)
        for (x2,y2,w2,h2) in goz_tespit2:
            cv.rectangle(frame_copy,(x2,y2),(x2+w2,y2+h2),(0,255,0),5)
    return frame_copy

while True:
    ret,frame = cap.read()
    frame = cv.flip(frame,1)
    
    if not ret:
        break
    res = tespit(frame)
    cv.imshow('GOZ TESPITI',res)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
