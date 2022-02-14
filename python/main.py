import sys
from time import time

import cv2
import numpy as np

import yolov5model

print('|-------------- STARTING --------------|')

# load model
model = yolov5model.YOLOv5Model('../yolomt/weights/last.pt')

# open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print('ERROR: Failed to open Video Capture Device!')
	sys.exit()

while True:
	# get new frame
	retval, frame = cap.read()
	originalDimensions = frame.shape

	# resize the image to model's training size
	frame = cv2.resize(frame, (640, 640))

	# detection and plotting
	start_t = time()
	modelOutput = model.detect(frame)
	frame = model.plotBoxes(modelOutput, frame, [(0, 255, 0), (255, 0, 0)])
	stop_t = time()

	# resize image back to original size
	frame = cv2.resize(frame, (originalDimensions[1], originalDimensions[0]))

	# calculate frames per second
	fps = 1/(stop_t - start_t)

	# draw frames per second on the frame
	cv2.putText(frame, 'FPS: {}'.format(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

	# show the results
	cv2.imshow('yolov5model', frame)

	if cv2.waitKey(5) & 0xFF == 27:
		break

# close the webcam
cap.release()

print('|---------------- DONE ----------------|')
