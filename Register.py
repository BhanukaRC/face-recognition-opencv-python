import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('Classifiers/face.xml')

def register(userid,name,secret):
                conn = sqlite3.connect("Face-Recognition.db")
                cmd="SELECT * FROM userDetails WHERE ID="+str(userid)
                cursor=conn.execute(cmd)
                isRecordExist=0
                for row in cursor:
                        isRecordExist=1
                if(isRecordExist==1):
                        #update name and secret if data exists
                        cmd="UPDATE userDetails SET Name="+str(name)+",Secret="+str(secret)+"WHERE ID="+str(userid)
                else:
                        #otherwise insert data
                        cmd="INSERT INTO userDetails(ID,Name,Secret) Values("+str(userid)+","+str(name)+","+str(secret)+")"                      
                conn.execute(cmd)
                conn.commit()
                conn.close()		

i=0
offset=50
print("Welcome!")

#taking user details
userid=input('enter your id : ')
name=input('enter your name : ')
secret=input('enter your secret : ')

#calling the function
register(userid,name,secret)

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        #pictures being stored
        cv2.imwrite("dataSet/face-"+userid +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.waitKey(100)
    if i>20:
        #20 pics will be stored in the dataset
        cam.release()
        cv2.destroyAllWindows()
        break
