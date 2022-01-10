import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from PIL import Image,ImageOps

import ssl,os,time

if(not os.environ.get('PYTHONHTTPSVERIFY','')and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context=ssl._create_unverified_context

X = np.load('image.npz')
X = X['arr_0'][:, :28]
y = pd.read_csv("labels.csv")
y = y["labels"]
print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

cam=cv2.VideoCapture(0)

while(True):
    try:
        ret,frame=cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        h,w= gray.shape
        upperLeft=(int(w/2-50),int(h/2-50))
        lowerRight=(int(w/2+50),int(h/2+50))
        cv2.rectangle(gray,upperLeft,lowerRight,(0,255,0))

        roi=gray[upperLeft[1]:lowerRight[1],upperLeft[0]:lowerRight[0]]

        pilImage=Image.fromarray(roi)

        image_bw=pilImage.convert('L')
        image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)

        image_bw_resized_inverted= ImageOps.invert(image_bw_resized)
        pxlFil=20
        minPxl=np.percentile(image_bw_resized_inverted,pxlFil)

        image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-minPxl,0,255)

        maxPxl=np.max(image_bw_resized_inverted)

        image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/maxPxl

        test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        
        test_predict=clf.predict(test_sample)
        print("the prediction is:",test_predict)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e :
        print(e)

cam.release()
cv2.destroyAllWindows()





