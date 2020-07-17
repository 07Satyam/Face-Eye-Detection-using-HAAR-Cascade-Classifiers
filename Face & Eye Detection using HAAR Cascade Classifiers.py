#Import Libraries 
import numpy as np
import cv2
import os
os.chdir('E:\Github Repo\Face & eye detection using Haar Cascade classifier')

# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load our image then convert it to grayscale
image = cv2.imread('Sushant.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI(region of interest) of the detected face as a tuple 
# It stores the top left coordinate and the bottom right coordiantes
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No faces found")
    
# We iterate through our faces array and 
# draw a rectangle over each face in faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('Eye Detection',image)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()    