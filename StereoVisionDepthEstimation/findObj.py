import cv2
import mediapipe as mp
import math


class ObjDetector:
    def __init__(self, mode=False):

        self.mode = mode

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

        self.mp_objectron = mp.solutions.objectron
        self.mp_draw = mp.solutions.drawing_utils
        self.objectron = self.mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Camera')

    def find(self,image,draw=True):

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.objectron.process(imgRGB)
        allObj = []
        h, w, c = image.shape
        if self.results.detected_objects:
            for detected_object in zip(self.results.detected_objects):
                self.mp_draw.draw_landmarks(image, detected_object.landmarks_2d, self.mp_objectron.BOX_CONNECTIONS)
                self.mp_draw.draw_axis(image, detected_object.rotation, detected_object.translation)
                myObj = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []

                for id, lm in enumerate(detected_object.landmarks_3d.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myObj["lmList"] = mylmList
                myObj["bbox"] = bbox
                myObj["center"] = (cx, cy)
                allObj.append(myObj)

        return allObj, image


