import os
import numpy as np
import cv2
from PIL import  Image

recognizer=cv2.face.LBPHFaceRecognizer_create();
path="faceData"
def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        facesImage=Image.open(imagePath).convert('L')
        faceNP=np.array(facesImage,'uint8')
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNP)
        IDs.append(ID)
        cv2.imshow("Dodavanje lica za ucenje",faceNP)
        cv2.waitKey(10)
    return np.array(IDs),faces
Ids,faces=getImageWithID(path)
recognizer.train(faces,Ids)
recognizer.save("faceLearn/learned.yml")
cv2.destroyAllWindows()
