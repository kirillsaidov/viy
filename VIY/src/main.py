# system packages
import sys
import time
import imutils
import datetime

# computer vision
import cv2
import cvui

# data processing
import pandas as pd

# viy help funcs package
import model.help_funcs as hf

def viy_launch():
    # initialize
    cvui.init(hf.VIY_WINDOW_ID)
    configs = hf.viy_setup()
    fwidth = configs['default_frame_width']
    fheight = configs['default_frame_height']
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
    vidwriter = cv2.VideoWriter('../results/vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vfps, (fwidth, fheight))

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
    
    # visuals
    button_name = 'START'
    button_start = False
    draw_age = [True]
    draw_info = [True]
    draw_gender = [True]
    draw_pedestrian_bb = [True]
    draw_settings = [True]
    __trackerMemoryDuration_ms = [tracker.trackerMemoryDuration_ms/1000]
    __trackerMaxDistance = [tracker.maxDistance]
    while True:
        # get the frame from video
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS counter
        total_frames, fps, fps_end_time = hf.fps(fps_start_time, total_frames)

        # tracking, age/gender estimation
        if button_start:
            # get pedestrian boudning boxes
            frame, list_pedestrian_coords = hf.getPedestrianCoords(model_pedestrian, frame, frame_size = (fwidth, fheight))

            # update all tracker coordinates
            # the object value contains a tuple of two values (objectId, bounding box data)
            objects = tracker.update(list_pedestrian_coords)

            # process objects in the current frame and compare them with all object ids in the existing list
            list_object_id, lpc_count, opc_count, dict_list_face = hf.processTrackerObjects(frame, objects, list_object_id, (
                model_face,                                                     # FACE
                model_age, transformer_age, classes_age, list_age,              # AGE
                model_gender, transformer_gender, classes_gender, list_gender   # GENDER
            ), draw_pedestrian_bb[0], draw_age[0], draw_gender[0], dict_list_face)

            # the time that goes by
            now = datetime.datetime.now()

            # collect the data into list (must be hour => minute for testing purposes)
            if current_time.minute < now.minute:
            # if current_time.hour < now.hour:
                # convert to dataframe and clear dict_list_face
                df = pd.DataFrame(dict_list_face.values())
                dict_list_face.clear()

                if not df.empty:
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
        else:
            frame = cv2.resize(frame, (fwidth, fheight))

        # ----------- GUI input ----------- #
        # INFO window
        cvui.window(frame, 0.001*fwidth, 0.001*fwidth, 0.2*fwidth, 0.175*fheight if draw_info[0] else 0.075*fheight, 'INFO')
        cvui.checkbox(frame, 0.01*fwidth, 0.045*fheight, 'Display settings', draw_settings)
        if draw_info[0]:
            lpc_txt =  "LPC:  {}".format(lpc_count)
            opc_txt =  "OPC:  {}".format(opc_count)
            fps_text = "FPS:  {:.2f}".format(fps)
            show_time = fps_end_time.strftime("TIME: %H:%M:%S")
            cvui.text(frame, 0.01*fwidth, 0.085*fheight, lpc_txt)
            cvui.text(frame, 0.01*fwidth, 0.105*fheight, opc_txt)
            cvui.text(frame, 0.01*fwidth, 0.125*fheight, fps_text)
            cvui.text(frame, 0.01*fwidth, 0.145*fheight, show_time)
        
        # start/stop video processing and recording
        button_state = cvui.button(frame, 0.12*fwidth, 0.035*fheight, button_name)
        if button_state and button_start:
            button_name = 'START'
            button_start = not button_start
        elif button_state and not button_start:
            button_name = 'STOP'
            button_start = not button_start
        
        # SETTINGS window
        if draw_settings[0]:
            cvui.window(frame, 0.001*fwidth, 0.18*fheight, 0.2*fwidth, 0.38*fheight, 'SETTINGS');

            # trackbars
            cvui.text(frame, 0.03*fwidth, 0.22*fheight, 'MEMORY DURATION (secs)')
            cvui.trackbar(frame, 0.001*fwidth + 3, 0.24*fheight, 0.18*fwidth, __trackerMemoryDuration_ms, 0, 10);
            cvui.text(frame, 0.041*fwidth, 0.32*fheight, 'MIN PROC. DISTANCE')
            cvui.trackbar(frame, 0.001*fwidth + 3, 0.34*fheight, 0.18*fwidth, __trackerMaxDistance, 0, 100);

            # checkboxes
            cvui.checkbox(frame, 0.01*fwidth, 0.43*fheight, 'Draw info', draw_info)
            cvui.checkbox(frame, 0.01*fwidth, 0.46*fheight, 'Display age data', draw_age)
            cvui.checkbox(frame, 0.01*fwidth, 0.49*fheight, 'Display gender data', draw_gender)
            cvui.checkbox(frame, 0.01*fwidth, 0.52*fheight, 'Display pedest. bb.', draw_pedestrian_bb)

        cvui.update()

        # update variables
        tracker.trackerMemoryDuration_ms = __trackerMemoryDuration_ms[0] * 1000
        tracker.maxDistance = __trackerMaxDistance[0]
        # --------------------------------- #        

        # show the video with results
        cv2.imshow(hf.VIY_WINDOW_ID, frame)
        
        # save as video
        if button_start:
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



# launch viy
viy_launch()