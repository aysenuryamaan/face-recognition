import cv2
import os
import numpy as np
import faceRecognition as fr

 
#This module takes images  stored in diskand performs face recognition
test_img=cv2.imread(' ')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)
   
    
   
for(x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(250,0,0),thickness=3)
         
# =============================================================================
# resized_img=cv2.resize(test_img,(600,800))
# cv2.imshow("face detection ",resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows
# =============================================================================
  
#Comment belows lines when running this program second time.Since it saves training.yml file in directory
faces,faceid=fr.labels_for_training_data('trainingimages')
face_recognizer=fr.train_classifier(faces,faceid)
face_recognizer.save('trainingdata.yml')

#Uncomment below line for subsequent runs
# =============================================================================
# face_recognizer=cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read('trainingdata.yml')
# =============================================================================

name={0:"name1", 1:"name2"}#creating dictionary containing names for each label

  
for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>23):       
        fr.put_text(test_img,predicted_name,x,y)
      
resized_img=cv2.resize(test_img,(700,800))
cv2.imshow("face detecetion",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
 
     

