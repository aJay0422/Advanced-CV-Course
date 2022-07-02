import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

# load header images
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

while True:
    # 1. Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)

    # 3.Check which fingers are up

    # 4. If Selection mode - Two fingers are up

    # 5.If Drawing mode - Index finger is up

    img[0:124, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)