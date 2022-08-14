# system packages
import sys
import time
import imutils
import datetime

# computer vision
import cv2

# data processing
import pandas as pd

# viy help funcs package
import model.help_funcs as hf

def main():
    # initialize
    configs = hf.viy_setup()
    model_face = configs['model_face']
    model_pedestrian = configs['model_pedestrian']
    tracker = configs['tracker']
    model_age = configs['model_age']
    classes_age = configs['classes_age']
    transformer_age = configs['transformer_age']
    model_gender = configs['model_gender']
    classes_gender = configs['classes_gender']
    transformer_gender = configs['transformer_gender']

    # open video stream cap
    cap = cv2.VideoCapture(configs['video_file'])

    # get cap properties for VideoWriter
    vwidth, vheight, vfps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

    # create a video writer to save the result
    vidwriter = cv2.VideoWriter('../results/vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vfps, (vwidth, vheight))

    # check if cap is opened
    if not cap.isOpened():
        print('ERROR: Failed to open Video Capture Device!')
        sys.exit()

    # FPS values
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    # Values for counting LPC - live person counter, OPC - overall person counter
    lpc_count = 0
    opc_count = 0
    opc_count_previous_hour = 0
    list_age = []
    list_gender = []
    list_object_id = []
    dict_list_face = {}

    # CURRENT TIME
    current_time = datetime.datetime.now()

    # DATA to store the [date, hour, #people, #males, #females, average_age]
    data = []
    
    # visualize bounding boxes
    draw_age = True
    draw_info = True
    draw_gender = True
    draw_pedestrian_bb = True
    while True:
        # get the frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # get pedestrian boudning boxes
        frame, list_pedestrian_coords = hf.getPedestrianCoords(model_pedestrian, frame, frame_size = None)

        # update all tracker coordinates
        # the object value contains a tuple of two values (objectId, bounding box data)
        objects = tracker.update(list_pedestrian_coords)

        # process objects in the current frame and compare them with all object ids in the existing list
        list_object_id, lpc_count, opc_count, dict_list_face = hf.processTrackerObjects(frame, objects, list_object_id, (
            model_face,                                                     # FACE
            model_age, transformer_age, classes_age, list_age,              # AGE
            model_gender, transformer_gender, classes_gender, list_gender   # GENDER
        ), draw_pedestrian_bb, draw_age, draw_gender, dict_list_face)

        # FPS counter
        total_frames, fps, fps_end_time = hf.fps(fps_start_time, total_frames)

        # the time that goes by
        now = datetime.datetime.now()

        # collect the data into list (must be hour => minute for testing purposes)
        if current_time.minute < now.minute:
        # if current_time.hour < now.hour:
            # convert to dataframe and clear dict_list_face
            df = pd.DataFrame(dict_list_face.values())
            dict_list_face.clear()

            # remove 'unknown' category
            df.drop(df[df[0] == 'unknown'].index, inplace = True)

            # most frequent age category
            freq_age = df[0].value_counts().idxmax() # FIXME: value_counts may be empty, check it before doing idxmax

            # count males/females
            gender_counts = df[1].value_counts()
            if len(gender_counts) != 2:
                if gender_counts.index.tolist()[0] == 'male':
                    males_count = gender_counts['male']
                    females_count = 0
                else:
                    males_count = 0
                    females_count = gender_counts['female']
            else:
                males_count = gender_counts['male']
                females_count = gender_counts['female']

            timedata = current_time.time().replace(second = 0, microsecond = 0, minute = current_time.minute, hour = current_time.hour)
            # timedata = current_time.time().replace(second = 0, microsecond = 0, minute = 0, hour = current_time.hour)
            data.append([current_time.date().strftime('%d/%m/%Y'), timedata.strftime('%M') , opc_count - opc_count_previous_hour, males_count, females_count, freq_age])
            # data.append([current_time.date().strftime('%d/%m/%Y'), timedata.strftime('%H') , opc_count - opc_count_previous_hour, males_count, females_count, freq_age])
            opc_count_previous_hour = opc_count
            current_time = now

        # show the time on frame/video

        # draw text
        if draw_info:
            lpc_txt = "LPC: {}".format(lpc_count)
            opc_txt = "OPC: {}".format(opc_count)
            fps_text = "FPS: {:.2f}".format(fps)
            show_time = fps_end_time.strftime("%H:%M:%S")

            hf.drawText(frame, lpc_txt, (5, 60), hf.YELLOW)
            hf.drawText(frame, opc_txt, (5, 90), hf.YELLOW)
            hf.drawText(frame, fps_text, (5, 30), hf.YELLOW)
            hf.drawText(frame, show_time, (5, 120), hf.YELLOW)

        # show the video with results
        cv2.imshow('VIY', frame)
        # vidwriter.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vidwriter.write(frame)

        # exit when ESCAPE key is presed or X button
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # close the webcam
    cap.release()

    # destroy all windows (release memory)
    cv2.destroyAllWindows()

    # close video writer
    vidwriter.release()

    # safe all data
    hf.saveData(data)


# execute main code
main()
