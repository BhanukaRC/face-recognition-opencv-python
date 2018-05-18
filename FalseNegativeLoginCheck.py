import cv2,os
import numpy as np
from PIL import Image 
import pickle
import sqlite3

#training a model
import trainer

recognizer = cv2.face.LBPHFaceRecognizer_create()

#traiined model
recognizer.read('trainer/trainer.yml')

cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

#function to get user details from database
def getUser(userid):
    conn=sqlite3.connect("Face-Recognition.db")
    cmd="SELECT * FROM userDetails WHERE ID="+str(userid)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
 
cam = cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        predicted_id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)

        #conf -> confidence level
        #higher the match <-> lower the confidence level
        #confidence level is too low -> it has to detect a 'perfect match'
        #so -> false-negativity achieved
        if(conf<73):
            profile=getUser(predicted_id)
            cv2.putText(im,str(conf),(x,y+h),font,0.5,(0,255,0))
            cv2.putText(im,str(profile[1]),(x,y+h+30),font,0.5,(0,255,0))
            cv2.putText(im,str(profile[2]),(x,y+h+60),font,0.5,(0,255,0))

	#not a match
        elif(conf>90):
            nbr_predicted = "Not recognized"
            cv2.putText(im,str(conf),(x,y+h),font,0.5,(0,255,0))
            cv2.putText(im,str(nbr_predicted),(x,y+h+30),font,1,(0,255,0))                
	#being detected and recognized
        #since not accepeted because of false-negativity
        else:
            nbr_predicted = "Confusing"
            cv2.putText(im,str(conf),(x,y+h),font,0.5,(0,255,0))
            cv2.putText(im,str(nbr_predicted),(x,y+h+30),font,1,(0,255,0))
        cv2.imshow('im',im)
        cv2.waitKey(10)
        if(cv2.waitKey(1)==ord('q')):
            exit();









