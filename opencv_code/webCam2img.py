import cv2
import numpy as np
import os
# from flask import Flask, request, jsonify
#
#
# app = Flask(__name__,static_folder='outputs')

img_path = "./data/JPEGImages/"
anno_path = "./data/Annotations/"

i=0

TrDict = {'csrt' : cv2.TrackerCSRT_create,
          'kcf' : cv2.TrackerKCF_create,
          'boosting' : cv2.TrackerBoosting_create,
          'mil' : cv2.TrackerMIL_create,
          'tld': cv2.TrackerTLD_create,
          'medianflow':cv2.TrackerMedianFlow_create,
          'mosse':cv2.TrackerMOSSE_create}

trackers = cv2.MultiTracker_create()


cap = cv2.VideoCapture(0)

ret, frame = cap.read()

for i in range(3):
    cv2.imshow('Frame',frame)
    # cv2.imshow('mask',mask)
    roi = cv2.selectROI('Frame', frame)
    tracker_i = TrDict['mosse']()
    trackers.add(tracker_i, frame, roi)
cv2.destroyAllWindows()



while True:
    if not ret:
        break

    ret, frame = cap.read()
    mask = np.zeros(frame.shape[:2], np.uint8)
    (success,rois) = trackers.update(frame)



    for roi in rois:

        (x,y,w,h) = [int(a) for a in roi] #좌표
        rect = (x,y,x+w,y+h)

        #cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,255),0) #frame에 사각형 roi 박스
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1) #마스크에 사각형박스




    #이미지 저장
    cv2.imwrite("./data/JPEGImages/" + '{0:05d}'.format(i) + ".jpg", frame)
    cv2.imwrite("./data/Annotations/" + '{0:05d}'.format(i) +".png",mask)

    #모델

    img_dir = os.path.join('./data/JPEGImages/', '%05d.jpg' % (i))
    frame = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    cv2.imshow('cam_load', frame)
    os.remove(img_dir)
    i += 1

    #press esc
    k = cv2.waitKey(24) #& 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        break











