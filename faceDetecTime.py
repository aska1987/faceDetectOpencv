import numpy as np
import cv2 as cv
import os
import time
import cv2
#from mtcnn.mtcnn import MTCNN
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
for im in os.listdir("./imageTest/"):
    print (im)
    nameim = os.path.basename(im)
    img = cv.imread("./imageTest/" + im)
    #gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    start_time = time.time()
    face_cascade.detectMultiScale(gray,1.2,3)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
    end_time = time.time()
    print ('total runtime: %f ms' %((end_time - start_time)))
    cv.imshow('img_' + nameim,img)
    cv.imwrite('Detected.jpg',img)
    cv.waitKey()
    cv.destroyAllWindows()
