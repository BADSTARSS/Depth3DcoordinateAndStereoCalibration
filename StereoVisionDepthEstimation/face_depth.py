import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import calibration

cap = cv2.VideoCapture(2)
left = cv2.VideoCapture(1)
detector = FaceMeshDetector(maxFaces=1)
detector_ = HandDetector(detectionCon=0.8, maxHands=1)
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
while True:
    success, img = cap.read()
    success_, left_ = left.read()
    img,faces = detector.findFaceMesh(img)
    hands_ = detector_.findHands(left_, draw=False)
    left_, img = calibration.undistortRectify(left_, img)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
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
        print(d)

        cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)
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
        cvzone.putTextRect(left_, f'{int(distanceCM)} cm', (x + 5, y - 10))



    cv2.imshow("Image", img)
    cv2.imshow("left", left_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)
cap.release()
left.release()

cv2.destroyAllWindows()