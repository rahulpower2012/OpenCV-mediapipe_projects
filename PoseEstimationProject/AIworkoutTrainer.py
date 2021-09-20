import time
import cv2
import PoseModule as pm
import numpy as np

cap = cv2.VideoCapture("TestVideos/1.mp4")
pTime = 0
detector = pm.PoseDetector()
count = 0
dir = 0
while True:
    success, img = cap.read()
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 293), (0, 100))
        bar = np.interp(angle, (210, 293), (450, 100))
        # print(str(int(per))+"%")

        # check for the dumbell curls
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        cv2.rectangle(img, (50, 100), (100, 450), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(bar)), (100, 450), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(count)}', (50, 400), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
