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
import model.yolov5model
import model.cnnclassifier as cnn

# colors
RED     = (0, 0, 255)
GREEN   = (0, 255, 0)
BLUE    = (255, 0, 0)
BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
YELLOW  = (0, 255, 255)
CYAN    = (255, 255, 0)
PURPUR  = (255, 0, 255)

"""
Model configuration setup

Returns: dict of configs
"""
def viy_setup():
    pass

"""
Model configuration setup in GUI mode

Returns: dict of configs
"""
def viy_gui_setup():
    pass

"""
Finds pedestrians on a frame and extracts their coordinates

Params:
    model_pedestrian = yolov5 pedestrian model
    frame = video/capture frame
    frame_size = resize to frame size; default: (w = 640, h = 640)

Returns: (frame, list of pedestrian coordinates)
"""
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

"""
Tracks and counts pedestrians on the frame

Params:
    frame = video/capture frame
    objects_in_frame = pedestrian coordinates and their ids
    objects_id_list = all pedestrian ids currently in the frame
    age_gender_info = pedestrian age and gender info 
    draw_pedestrian_bb = draw a bounding box around the pedestrian object; default: True
    draw_age = display pedestrian age information; default: True
    draw_gender = display pedestrian gender information; default: True
    dict_list_face = gender and age information tied to a particular face

Returns: (objects_id_list, lpc_count, opc_count, dict_list_face)
"""
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

"""
Predicts pedestrian age and gender

Params:
    frame = video/capture frame
    bbox = bounding box face coordinates
    objectId = object id assigned 
    dict_list_face = gender and age information tied to a particular face 
    age_gender_info = model_face, model_age, transformer_age, classes_age, list_age, model_gender, transformer_gender, classes_gender, list_gender

Returns: dict_list_face
"""
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

"""
Detects a face on a frame

Params:
    model = face yolov5 model
    frame = video/capture frame
    bbox = bounding box face coordinates

Returns: a cropped face image
"""
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

"""
Calculates FPS

Params:
    fps_start_time = frame start time
    total_frames = frames passed

Returns: total_frames, fps, fps_end_time
"""
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

"""
Displays text on frame surface
"""
def drawText(frame, text, position = (10, 10), color = RED, fontScale = 1, thickness = 1, lineType = cv2.LINE_AA):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, thickness, lineType)

"""
Saves VIY data as EXCEL, CSV files

Params:
    data = VIY data
    filename = save file name
    csv = save to csv; default: True
    excel = save to excel; default: True

Returns: total_frames, fps, fps_end_time
"""
def saveData(data, filename = 'viy_summary', csv = True, excel = True):
    df = pd.DataFrame(data, columns = ['date', 'hour', '#people', '#males', '#females', 'age'])

    if excel:
        saveAsExcel(df, '../results/' + filename + '.xlsx')
    
    if csv:
        saveAsCSV(df, '../results/' + filename + '.csv')

"""
Saves dataframe to excel file
"""
def saveAsExcel(df, filename = 'viy_summary.xlsx'):
    df.to_excel(filename, index = False)

"""
Saves dataframe to csv file
"""
def saveAsCSV(df, filename = 'viy_summary.csv'):
    df.to_csv(filename, index = False, sep = ';')



