import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


brushThickness = 15
eraserThickness = 50
menuPath = "Menu"
myList = os.listdir(menuPath)
overLayList =[]
for imPath in myList:
    image = cv2.imread(f'{menuPath}/{imPath}')
    overLayList.append(image)

initMenu = overLayList[0]
colorCur = (255, 0, 255)

cap = cv2.VideoCapture(3)

cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp =0,0
imgCanvas = np.zeros((720,1280,3), np.uint8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:
        #print(lmList)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        #print(fingers)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img,(x1, y1-25), (x2, y2+25), colorCur, cv2.FILLED)
            #print("Selection mode")
            if y1<125:
                if 220<x1<400:
                    initMenu = overLayList[0]
                    colorCur = (255,0,255)
                    xp, yp =0,0
                elif 470<x1<650:
                    initMenu = overLayList[1]
                    colorCur = (255, 0, 0)
                    xp, yp = 0, 0
                elif 730<x1<910:
                    initMenu = overLayList[2]
                    colorCur = (0, 255, 0)
                    xp, yp = 0, 0
                elif 1000<x1<1150:
                    initMenu = overLayList[3]
                    colorCur = (0, 0, 0)
                    xp, yp = 0, 0

        if not (fingers[1] and fingers[2]):
            cv2.circle(img, (x1, y1), 15, colorCur,cv2.FILLED)
            #print("Drawing mode")
            if xp==0 and yp==0:
                xp, yp = x1, y1

            if colorCur ==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), colorCur, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), colorCur, eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1), colorCur, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), colorCur, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = initMenu
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image",img)
    cv2.imshow("ImCanvas", imgCanvas)
    cv2.imshow("InvImg", imgInv)
    cv2.waitKey(1)