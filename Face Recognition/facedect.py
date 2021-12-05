import cv2 as cv
import numpy as np
import random
import time

start_time = time.time()
fixed_time = 10
timing = 0

# Adjusting the size of the frame
def rescaleFrame(frame , scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width , height)
    return cv.resize(frame , dimensions , interpolation= cv.INTER_AREA)

#Taking Video Input from the camera
capture = cv.VideoCapture(0)

isTrue , frame =  capture.read()
img = rescaleFrame(frame)
ha = int(img.shape[0])
wa = int(img.shape[1])


fourcc = cv.VideoWriter_fourcc(*"MJPG")
out = cv.VideoWriter('vv.avi', fourcc , 6.0 , (wa,ha))

wa = int(wa * 0.75)
ha = int(ha * 0.75)

people = ['SB' , 'NB' , 'YB']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


c = 0

while (fixed_time > timing):

    isTrue , frame =  capture.read()
    elapsed_time = time.time()

    timing = int(elapsed_time - start_time)

 

    if not isTrue:
        break

    img = rescaleFrame(frame)

    gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY) 

    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray , scaleFactor = 1.1 , minNeighbors=3)
    for (x,y,w,h) in faces_rect:
        cv.rectangle(img , (x,y) , (x+w,y+h) , (0,255,0) , 1)

    xa = []
    ya = []
    xb = []
    yb = []

    if(len(faces_rect)>1):
        for (x , y , w , h) in faces_rect:
            xa.append(x)
            ya.append(y)
            xb.append(x+w)
            yb.append(y+h)

        xa.sort()
        ya.sort()
        xb.sort()
        yb.sort()

        m=0
        n=0
    
        l=len(xa)
        cropped_image = img[ ya[0]-n:yb[l-1]+n , xa[0]-m:xb[l-1]+m]


    elif(len(faces_rect)==1):
        for (x , y , w , h) in faces_rect:
            m = 0
            n = 0
            cropped_image = img[ y-n:y+h+n , x-m:x+w+m]
    

    elif(len(faces_rect)==0):
        cropped_image = rescaleFrame(img , scale = 0.75)
   
    cropped_image = cv.resize(cropped_image , (wa , ha) , interpolation = cv.INTER_AREA)
    p = cv.cvtColor(cropped_image , cv.COLOR_BGR2GRAY)
    label , confidence = face_recognizer.predict(p)

    top = (img.shape[0] - cropped_image.shape[0])//2
    left = (img.shape[1] - cropped_image.shape[1])//2

    bordered = cv.copyMakeBorder(cropped_image , top , top ,left , left , borderType = cv.BORDER_CONSTANT , value = 0)
    out.write(bordered)


    cv.imshow('Detected Faces' , bordered)

    if(label>=0 and confidence>65):
        c=c+1
    

if(c>10):
    print(True)

else:
    print(False)

capture.release()
out.release()

cv.destroyAllWindows()