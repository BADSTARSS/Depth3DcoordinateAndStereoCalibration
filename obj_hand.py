import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
import mediapipe as mp
import StereoVisionDepthEstimation.calibration as calibration
import StereoVisionDepthEstimation.triangulation

# Webcam
left = cv2.VideoCapture(1)
right = cv2.VideoCapture(2)
left.set(3, 640)
left.set(4, 480)
right.set(3, 640)
right.set(4, 480)

mp_draw = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)
objectron = mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Camera')
# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C


def Caldist(x,x_):
    for i in range(0,len(x and x_)):
        x[i] = int(x[i])

        # if(x[i] > x_[i]):
        #     x[i]==x[i]
        #     abs_ = math.sqrt(x[i] ** 2 - x_[i] ** 2)
        #     return abs_
        if(x[i] > x_[i] ):
            abs_ = x[i]-x_[i]
            return abs_
        elif (x[i] < x_[i]):
            abs_ = x_[i]-x[i]
            return abs_
        elif (x[i] == x_[i]):
            abs_ = '0'
            return abs_
        # else:
        #     abs_ = math.sqrt(x_[i]**2 - x[i]**2)
        #     return abs_
        else :
            abs_ = x_[i]-x[i]
            return abs_


# Loop
while True:
    success, left_ = left.read()
    # hands_ = detector.findHands(left_, draw=False)
    hands_ = detector.findHands(left_, draw=False)
    success_, right_ = right.read()
    object_ = objectron.process(right_)
    ################## CALIBRATION ####################
    left_, right_ = calibration.undistortRectify(left_, right_)
    bboxs = []

    distance1 = []
    distance2 = []
    if hands_:
        lmList = hands_[0]['lmList']
        x, y, w, h = hands_[0]['bbox']
        x1, y1 ,z= lmList[5]
        x2, y2 ,z= lmList[17]

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM_ = A * distance ** 2 + B * distance + C

        # print(distanceCM, distance)

        cv2.rectangle(left_, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cvzone.putTextRect(left_, f'{int(distanceCM_)} cm', (x+5, y-10))
        distance1.append(distanceCM_)


    if object_.detected_objects:
        for detected_object in object_.detected_objects:
            mp_draw.draw_landmarks(right_, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_draw.draw_axis(right_, detected_object.rotation, detected_object.translation)
            bbox = []

            for id, lm in enumerate(detected_object.landmarks_3d.landmark):
                h, w, c = right_.shape
                x, y = int(lm.x * w), int(lm.y * h)
                bbox.append([x, y])

            bboxs.append(bbox)
            lmlist_ = bboxs[0]
            x, y = lmlist_[0]
            w, h = lmlist_[0]
            x1, y1 = lmlist_[5]
            x2, y2 = lmlist_[8]

            distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            A, B, C = coff
            distanceCM = A * distance ** 2 + B * distance + C
            # print(distanceCM)

            cv2.putText(right_, f'Objectpoint: {int(distanceCM)} cm', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)
            distance2.append(distanceCM)
    dist_ = Caldist(distance1 , distance2)
    # print(dist_)

    cv2.putText(right_, f'distancebetween: {(dist_)} cm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2)



    cv2.imshow("left_", left_)
    cv2.imshow("right_", right_)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


right.release()
left.release()

cv2.destroyAllWindows()
