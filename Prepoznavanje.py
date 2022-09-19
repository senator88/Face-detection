import numpy as np
import cv2
face_cascade=cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

capture=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('faceLearn/learned.yml')
id=0
font=cv2.FONT_HERSHEY_COMPLEX
while 1:
    ret,img=capture.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,10)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (id == 2):
            id = "Dino"
        if id == 1:
            id = "Vedran"
        if id == 3:
            id = "Test1"
        if id == 4:
            id = "Novi"
        if id == 5:
            id = "Test2"
        if id == 8:
            id = "R.Carlos"
        eyes = eye_cascade.detectMultiScale(gray, 1.2, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(img, id, (x, y), font,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
