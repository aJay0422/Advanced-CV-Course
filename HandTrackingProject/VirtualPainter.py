import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#########################
brushThickness = 15
eraserThickness = 100
#########################
folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

# load header images
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    # 1. Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        # 3.Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0   # reset starting points everytime we pause drawing
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            print("Selection Mode")

            # Checking for the click
            if y1 < 124:
                if 320 < x1 < 456:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 540 < x1 < 676:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 760 < x1 < 896:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 985 < x1 < 1149:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

        # 5.If Drawing mode - Index finger is up
        if fingers[1] and (fingers[2] == 0):
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    # create a mask the add canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:124, 0:1280] = header

    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("ImageCanvas", imgCanvas)
    cv2.waitKey(1)