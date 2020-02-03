import cv2
import numpy as np
from PIL import Image
import os
import pickle

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
image_data = []
skip = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
with open("labels.pickle" , "rb") as f:
    old_labels = pickle.load(f)
    labels = {v:k for k,v in old_labels.items()}


while True:
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , 1.3 , 5)
    for face in faces:
        skip  += 1
        (x , y , w, h) = face
        cv2.rectangle(frame , (x , y), (x+w , y+h) , (255,0,0) , 2)
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray , 1.3, 5)
        id_ , conf = recognizer.predict(roi_gray)                                         #################   prediction   ################
        if conf >= 45 and conf <= 85:
            cv2.putText(frame , labels[id_], (x,y) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255,255,255))
            #print(labels[id_])

        #print(conf)
        for eye in eyes:
            (ex , ey , ew , eh) = eye
            cv2.rectangle(roi_color , (ex , ey) , (ex+ew , ey+eh) , (0,255,0) , 2)

    cv2.imshow("face_recog" , frame)
    key = cv2.waitKey(1)
    if key & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()