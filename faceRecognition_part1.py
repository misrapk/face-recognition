import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('E:\Training Ducat\documents of ml\haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    #if face is there
    for (x,y,w,h) in faces:
        cropped_faces = img[y:y+h, x:x+w]

    return cropped_faces



cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        #save the faces
        file_name_path = "E:\\Training Ducat\\vs cod ml\\Face detection\\Face_data\\user" + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        #count number of imagees 
        cv2.putText(face,str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,100),2) 
        cv2.imshow('Face Cropper', face)
    else:
        # print("face not found!!!")
        pass

    if cv2.waitKey(1)==13 or count == 100:
        break

