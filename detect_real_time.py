import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir
from model.CNN import *

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	
print('[INFO] loading models...')

age_model = ageModel()
gender_model = genderModel()

#age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
output_indexes = np.array([i for i in range(0, 101)])

#------------------------

vc = cv2.VideoCapture(0) #capture webcam

while(True):
	ret, img = vc.read()
	img = cv2.resize(img, (640, 360))
	
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	for (x,y,w,h) in faces:
		if w > 130: #ignore small faces
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1) #draw rectangle to main image
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			
			try:
				margin = 30
				margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
				detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
			except:
				print("detected face has no margin")
			
			try:
				detected_face = cv2.resize(detected_face, (224, 224))
				
				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255
				
				#find out age and gender
				age_distributions = age_model.predict(img_pixels)
				age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))
				
				gender_distribution = gender_model.predict(img_pixels)[0]
				gender_index = np.argmax(gender_distribution)
				
				if gender_index == 0: 
					gender = "Female"
				else: 
					gender = "Male"
			
				#background for age gender declaration
				info_box_color = (255,0,0)
				#triangle_cnt = np.array( [(x+int(w/2), y+10), (x+int(w/2)-25, y-20), (x+int(w/2)+25, y-20)] )
				triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
				cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
				cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+120,y-90),info_box_color,cv2.FILLED)
				
				#labels for age and gender
				cv2.putText(img, age, (x+int(w/2)+80, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
				cv2.putText(img, gender, (x+int(w/2)-42, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
				
			except Exception as e:
				print("exception",str(e))
			
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
vc.release()
cv2.destroyAllWindows()