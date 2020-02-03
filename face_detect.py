import os
import numpy as np
from PIL import Image
import cv2
import pickle
import cv2


x_train = []
y_train = []
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir , "data")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
label_ids = {}
current_id = 1
recognizer = cv2.face.LBPHFaceRecognizer_create()


for root , dir , files in os.walk(data_dir):                        #here you enter the name of img file ,,, (like Brad pitt)
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jfif") or file.endswith(".png"):
            path = os.path.join(root , file)
            label = os.path.basename(root).replace(" " , "-").lower()
            #print(label)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            pil_image = Image.open(path).convert("L")
            resize_img = pil_image.resize((500,500) , Image.ANTIALIAS)
            image_array = np.array(resize_img , "uint8")
            faces = face_cascade.detectMultiScale(image_array , 1.3 , 5)
            #print(label)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_train.append(id_)

#print(x_train)
#print(y_train)


with open("labels.pickle" , "wb") as f:
    pickle.dump(label_ids , f)

recognizer.train(x_train , np.array(y_train))
recognizer.save("trainner.yml")