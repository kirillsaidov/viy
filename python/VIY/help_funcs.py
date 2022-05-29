# system packages
import sys
import time
import imutils
import datetime

# computer vision
import cv2
import torch

# data processing
import numpy as np
import pandas as pd

# custom packages
import yolov5model
import cnnclassifier as cnn

# colors
RED     = (0, 0, 255)
GREEN   = (0, 255, 0)
BLUE    = (255, 0, 0)
BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
YELLOW  = (0, 255, 255)
CYAN    = (255, 255, 0)
PURPUR  = (255, 0, 255)

def getPedestrianCoords(model_pedestrian, frame, frame_size = (640, 640)):
    # resize the frame
    if frame_size is not None:
        frame = imutils.resize(frame, width = frame_size[0], height = frame_size[1])

    # detect pedestrians
    modelOutput = model_pedestrian.detect(frame)

    # empty list of coordinates
    lst_pedestrian_coords = []

    # start X,Y -> center X,Y    |    end X,Y -> width, height
    coordinates = model_pedestrian.getBoxData(modelOutput, frame)
    for i in coordinates:
        startX = i[0]
        startY = i[1]
        endX = i[2]
        endY = i[3]
        label = model_pedestrian.classToString(i[4])

        # coordinate of list
        coor_list = [startX, startY, endX, endY]

        # append list coordinates to list
        lst_pedestrian_coords.append(coor_list)

    return frame, lst_pedestrian_coords

def processTrackerObjects(frame, objects_in_frame, objects_id_list, age_gender_info, draw_pedestrian_bb = True, draw_age = True, draw_gender = True, dict_list_face = None):
    # expand vars
    model_face, model_age, transformer_age, classes_age, list_age, model_gender, transformer_gender, classes_gender, list_gender = age_gender_info

    # if no objects were detected skip this part
    if objects_in_frame != None:
        for (objectId, bbox) in objects_in_frame.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # check if its large enough
            if y2 - y1 >= 180:
                # detect face, estimate age and gender
                if objectId not in dict_list_face or dict_list_face[objectId] == ['unknown', 'unknown']:
                    dict_list_face = predictAgeGender(frame, bbox, objectId, dict_list_face, age_gender_info)

                # list of objectId
                if objectId not in objects_id_list:
                    objects_id_list.append(objectId)

                if draw_pedestrian_bb:
                    # draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 1)

                    # show the text i.e., ID of object
                    text = "ID: {}".format(objectId)
                    cv2.rectangle(frame, (x1, y1), (x2, y1 - 15), GREEN, -1)
                    drawText(frame, text, (x1, y1 - 5), BLACK, thickness = 1, fontScale = 0.6)

                if draw_gender:
                    text = dict_list_face[objectId][1]
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 1)
                    drawText(frame, text, (x1, y1 + 7), GREEN, thickness = 1, fontScale = 0.6)

                if draw_age:
                    text = dict_list_face[objectId][0]
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 1)
                    drawText(frame, text, (x1, y1 + 17), GREEN, thickness = 1, fontScale = 0.6)

    # count the LPC and OPC
    lpc_count = 0 if objects_in_frame is None else len(objects_in_frame)
    opc_count = 0 if objects_id_list is None else len(objects_id_list)

    return objects_id_list, lpc_count, opc_count, dict_list_face

def predictAgeGender(frame, bbox, objectId, dict_list_face, age_gender_info):
    model_face, model_age, transformer_age, classes_age, list_age, model_gender, transformer_gender, classes_gender, list_gender = age_gender_info

    # detect and crop face from pedestrian bounding box data
    frame_face = detectFace(model_face, frame, bbox)
    
    # predict age and gender
    if frame_face is not None:
        frame_face = cnn.img2array(frame_face)
        pred_age = cnn.predict(model_age, frame_face, transformer_age, classes_age)
        pred_gender = cnn.predict(model_gender, frame_face, transformer_gender, classes_gender)

        dict_list_face.update({objectId: [pred_age, pred_gender]})
    else:
        dict_list_face.update({objectId: ['unknown', 'unknown']})

    return dict_list_face

def detectFace(model, frame, bbox):
    # expand bbox
    x1, y1, x2, y2 = bbox
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # crop pedestrian
    cframe = frame[y1:y2, x1:x2]

    # detect pedestrians
    modelOutput = model.detect(cframe)
    boxdata = model.getBoxData(modelOutput, cframe)
    if len(boxdata) != 0:
        x1, y1, x2, y2, label = boxdata[0]
        return cframe[y1:y2, x1:x2]
    
    return None

def fps(fps_start_time, total_frames):
    total_frames = total_frames + 1
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    
    fps = 0
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames/time_diff.seconds)

    return total_frames, fps, fps_end_time

def drawText(frame, text, position = (10, 10), color = RED, fontScale = 1, thickness = 1, lineType = cv2.LINE_AA):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, thickness, lineType)

def saveData(data):
    df = pd.DataFrame(data, columns = ['date', 'hour', '#people', '#males', '#females', 'age'])
    saveAsExcel(df, 'data_summary.xlsx')
    saveAsCSV(df, 'data_summary.csv')

def saveAsExcel(df, filename = 'data_summary.xlsx'):
    df.to_excel(filename, index = False)

def saveAsCSV(df, filename = 'data_summary.csv'):
    df.to_csv(filename, index = False, sep = ';')



