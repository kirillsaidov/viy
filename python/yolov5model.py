import random
import torch
import cv2

"""
YOLOv5 Model
"""
class YOLOv5Model:
    """
    Initializes the class
    """
    def __init__(self, path_to_weights, force_reload = False):
        # load model
        self.model = self.loadModel(path_to_weights, force_reload)
        self.classes = self.model.names

        # get device type: cpu or gpu (cuda)
        self.device = getDevice()

        print('Model has been loaded: {}'.format(path_to_weights))

    """
    Loads YOLOv5 model weights into the memory
    """
    def loadModel(self, path_to_weights, force_reload):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path = path_to_weights, force_reload = force_reload)
        return model

    """
    Detects objects on a frame
    """
    def detect(self, frame):
        # send to device and detect
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        # unpack data
        labels, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return (labels, coord)

    """
    Converts a numeric label value to a string
    """
    def classToString(self, id):
        return self.classes[int(id)]
    
    
    """
    Returns a list of tuples (x1, y1, x2, y2, label)
    """
    def getBoxData(self, modelOutput, frame, conf = 0.5):
        labels, coord = modelOutput
        x_size, y_size = frame.shape[1], frame.shape[0]

        # save data
        boxData = list()
        nlabels = len(labels)
        for i in range(nlabels):
            j = coord[i]

            # check confidence
            if j[4] >= conf:
                # save data
                x1, y1, x2, y2 = int(j[0]*x_size), int(j[1]*y_size), int(j[2]*x_size), int(j[3]*y_size)
                boxData.append((x1, y1, x2, y2, labels[i]))

        return boxData

    """
    Plots boudning boxes around detected objects
    """
    def plotBoxes(self, modelOutput, frame, colorBGR = None, conf = 0.5, thickness = 3):
        labels, coord = modelOutput
        x_size, y_size = frame.shape[1], frame.shape[0]

        # draw boxes
        nlabels = len(labels)
        ncolors = 0 if colorBGR is None else len(colorBGR)
        for i in range(nlabels):
            j = coord[i]

            # plot if our confidence if high
            if j[4] >= conf:
                # choose color
                color = None
                if ncolors != nlabels:
                    color = (0, 255, 0)
                else:
                    color = colorBGR[i]

                # plot
                x1, y1, x2, y2 = int(j[0]*x_size), int(j[1]*y_size), int(j[2]*x_size), int(j[3]*y_size)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, self.classToString(labels[i]), (x1, y1 - thickness - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

        return frame

"""
Returns the available device: cpu or gpu (cuda)
"""
def getDevice():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
