import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')
eyes_cascade=cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
capture=cv2.VideoCapture(0)
id=input('Unesite id korisnika -> ')
sampleN=0;
while 1:
    ret,img=capture.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,10)
    for(x,y,w,h) in faces:
        sampleN=sampleN+1;
        cv2.imwrite("faceData/User."+str(id)+"."+str(sampleN)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(25)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    if sampleN>20:
       break

capture.release()
cv2.destroyAllWindows()


