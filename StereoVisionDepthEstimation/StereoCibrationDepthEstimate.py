import cv2
import math
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe as mp
import StereoVisionDepthEstimation.calibration as calibration
import time
import distance as dis_

left = cv2.VideoCapture(1)
right = cv2.VideoCapture(2)
left.set(3, 640)
left.set(4, 480)
right.set(3, 640)
right.set(4, 480)
mp_draw = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

detector = FaceMeshDetector(maxFaces=1)
detector_ = HandDetector(detectionCon=0.8, maxHands=1)
objectron = mp_objectron.Objectron(static_image_mode=True,
                                   max_num_objects=5,
                                   min_detection_confidence=0.5,
                                   model_name='Camera')
# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
A, B, C = coff
# Loop
while True:

    success, left_ = left.read()
    success_, right_ = right.read()
    hands_ = detector_.findHands(left_, draw=False)
    object_ = objectron.process(right_)
    left_, faces = detector.findFaceMesh(left_)
    start = time.time()
    ################## CALIBRATION ####################
    left_, right_ = calibration.undistortRectify(left_, right_)
    ################## CALIBRATION ####################

    pointa = 0
    pointa_ = 0
    cx = 0
    cy = 0
    distanceCM_ = 0
    d = 0
    if hands_:
        lmList = hands_[0]['lmList']
        x, y, w, h = hands_[0]['bbox']
        x1, y1, z = lmList[5]
        x2, y2, z = lmList[17]

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C
        # print(distanceCM, distance)
        cv2.rectangle(left_, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cvzone.putTextRect(left_, f'Depth{int(distanceCM)} cm', (x + 5, y - 10))

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        centerpoint1 = dis_.findcenter(pointLeft, pointRight)
        pointa, pointa_ = centerpoint1
        # Drawing
        # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3

        # # Finding the Focal Length
        # d = 50
        # f = (w*d)/W
        # print(f)
        # Finding distance
        f = 840
        d = (W * f) / w
        # print(d)

        cvzone.putTextRect(left_, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

    if object_.detected_objects:
        for detected_object in object_.detected_objects:
            mp_draw.draw_landmarks(right_, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_draw.draw_axis(right_, detected_object.rotation, detected_object.translation)

            xList = []
            yList = []
            lmlist = []
            for id, lm in enumerate(detected_object.landmarks_2d.landmark):
                h, w, c = right_.shape
                px, py = int(lm.x * w), int(lm.y * h)
                lmlist_ = int(lm.x * w), int(lm.y * h)
                lmlist.append(lmlist_)
                xList.append(px)
                yList.append(py)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                x, y, w, h = bbox
                w_ = math.hypot(bbox[0] - bbox[2], bbox[1] - bbox[3])
                W = 6.3
                f = 840
                d = (W * f) / w_

            cv2.putText(right_, f'Objectpoint: {int(d)} cm', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

    Dbetween = dis_.finddist(pointa, pointa_, cx, cy, distanceCM_, d)
    distanceCM = A * Dbetween ** 2 + B * Dbetween + C
    # print(distanceCM)
    cv2.putText(right_, f'distanceFace-Obj: {int(distanceCM)} cm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 0), 2)
    cv2.putText(left_, f'distanceFace-Obj: {int(distanceCM)} cm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 0), 2)
    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
    cv2.putText(right_, f'FPS: {int(fps)}', (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(left_, f'FPS: {int(fps)}', (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("left_", left_)
    cv2.imshow("right_", right_)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

right.release()
left.release()

cv2.destroyAllWindows()
