import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
#########################################
wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
frameR =100
smoothening = 5
#########################################
cap = cv2.VideoCapture(3)
cap.set(3, wCam)
cap.set(4, hCam)

plocX, plocY = 0,0
clocX, clocY = 0,0

pTime = 0
cTime = 0

detector = htm.handDetector(maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1, x2, y2)

        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam - frameR),
                      (255, 0, 255), 2)
        #Moving
        if fingers[1]==1 and fingers[2]==0:
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr-3))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            #Smoothening
            clocX = plocX + (x3-plocX) /smoothening
            clocY = plocY + (y3-plocY) /smoothening


            autopy.mouse.move(clocX, clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX, plocY = clocX, clocY
        #Clicking
        if fingers[1] == 1 and fingers[2]==1:
            length, img, lineInfo = detector.findDistance(img,8,12)
            if length<20:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0 ),cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
