
#TODO: Train the data


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

#we need path for the training images 
data_path = 'E:\\Training Ducat\\vs cod ml\\Face detection\\Face_data\\'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

training_data, labels = [], []    #empty list

for i, files in enumerate(onlyfiles):      #enumerate provide enumeration
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #append the training data in to array
    training_data.append(np.asarray(images, dtype = np.uint8))
    labels.append(i)

#call labels
labels = np.asarray(labels, dtype= np.int32)

model = cv2.face.LBPHFaceRecognizer_create()     #linearbinaryphasehistogramfacerecognizer
model.train(np.asarray(training_data),np.asarray(labels))
print("Model Training Complete!")


#TODO: predict data
#give the path of face_classifier
face_classifier = cv2.CascadeClassifier('E:\Training Ducat\documents of ml\haarcascade_frontalface_default.xml')

#def a function for face detection
def face_detector(img,size=0.5):
    #convert image in grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #if face is not there
    if faces is():
        return img,[]
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,200,100), 2)

        #now we need region of interst
        roi = img[y: y+h, x:x+w]
        roi = cv2.resize(roi,(200,200))

    return img, roi

#now we need the main function

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    #pass frame through face detector
    image, face = face_detector(frame)

    #handle errors
    try :
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #predict the model
        result = model.predict(face)
        if result[1] < 500:
            confidence = int(100 * (1 - (result[1])/300))    #IT WILL GIVE THE CONFIDENCE PERCENTAGE
            print(confidence)
            display_string = str(confidence) + '%confidence it is user'    #display the confidence 
        cv2.putText(image, display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX, 1,(250,120,100),2)



        #DETECT THE FACE
        if confidence > 75:
            cv2.putText(image, "Unlocked",(250,450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Locked",(250,450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
            cv2.imshow('Face Cropper', image)




    except:       #this will handle if there is no face
        cv2.putText(image, "Face not found!",(250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,100),2)

        cv2.imshow("Face cropper", image)
        pass

    if cv2.waitKey(1) ==13:
        break

cap.release()
cv2.destroyAllWindows()