import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy
import math

wCam, hCam = 640, 480
frameR = 100
smoothening = 6
plocx,plocy = 0,0
clocx,clocy = 0,0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
detector = htm.handDetector(maxHands=1)
wscr,hscr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    #print(wscr,hscr)
    # 2. Get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1,y1,x2,y2)

        # 3. Check which fingers are up
        fingers = detector.fingerup()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 3)

        # 4. Only Index Finger : Moving Mode
        if fingers[1]==1 and fingers[2]==0:
            # 5. Convert Coordinates
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wscr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hscr))

            # 6. Smoothen Values
            clocx = plocx+(x3-plocx)/smoothening
            clocy = plocy+(y3-plocy)/smoothening

            # 7. Move Mouse
            autopy.mouse.move(wscr-x3,y3)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocx,plocy = clocx,clocy
        # 8. Both Index and Middle fingers are up : Clicking Mode
        if fingers[1]==1 and fingers[2]==1:
            # 9. Find distance between fingers
            length,img,lineInfo = detector.findDistance(8,12,img)
            print(length)
            # 10. Click mouse if distance short
            if length<40:
                cv2.circle(img, (lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    # 12. Display
    cv2.imshow("Image",img)
    cv2.waitKey(1)



