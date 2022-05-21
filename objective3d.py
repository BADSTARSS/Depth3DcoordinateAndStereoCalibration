import cv2
import mediapipe as mp
import StereoVisionDepthEstimation.distance as dis_
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

#
#       3 + + + + + + + + 7
#       +\                +\          UP
#       + \               + \
#       +  \              +  \        |
#       +   4 + + + + + + + + 8       | y
#       +   +             +   +       |
#       +   +             +   +       |
#       +   +     (0)     +   +       .------- x
#       +   +             +   +        \
#       1 + + + + + + + + 5   +         \
#        \  +              \  +          \ z
#         \ +               \ +           \
#          \+                \+
#           2 + + + + + + + + 6

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

objectron = mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Camera')

cap = cv2.VideoCapture(1)

while cap.isOpened():
  success, image = cap.read()
  object_ = objectron.process(image)
  bboxs = []
  if object_.detected_objects:
    for detected_object in object_.detected_objects:
      mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
      mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

      xList = []
      yList = []
      lmlist = []
      for id, lm in enumerate(detected_object.landmarks_2d.landmark):
        h, w, c = image.shape
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
        print(bbox)
      x1, y1  = lmlist[4]
      x2, y2  = lmlist[5]
      distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
      A, B, C = coff
      distanceCM_ = A * distance ** 2 + B * distance + C
      cv2.rectangle(image, (x, y), (x, y), (255, 0, 255), 3)
      cv2.putText(image, f'Objectpoint: {int(distanceCM_)} cm', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (255, 0, 0), 2)

  cv2.imshow('MediaPipe Objectron', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
cap.release()