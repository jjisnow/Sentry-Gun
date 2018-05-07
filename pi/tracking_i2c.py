# importing the neccessery libraries
import argparse

import cv2
import smbus

bus = smbus.SMBus(1)

address = 0x04


def writenum(value):
    bus.write_byte(address, value)
    return -1


# argument parser with minimum area for it to pick up as motion
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=200, help="minimum area")
args = vars(ap.parse_args())

# getting the video out from the webcam
cap = cv2.VideoCapture(0)

# getting the background subtractor ready for use
fgbg = cv2.createBackgroundSubtractorMOG2()

while (1):
    # starting the loop while  reading from the video capture
    ret, frame = cap.read()

    # applying the bakground subtractor
    fgmask = fgbg.apply(frame)

    thresh = fgmask
    thresh = cv2.GaussianBlur(thresh, (21, 21), 0)
    #	thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # making the image binary and adjusting it for the contouring
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # find the contours aroung the edges of the motion
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # putting the contour area through the arguement parser for minimum area
        c = max(cnts, key=cv2.contourArea)

        # draw the rectangle around the object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(frame, (x + w // 2, y + h // 2), 2, (0, 255, 0), -1)

        a = (x + w) / 2
        writenum(a)
        print(a)

        b = (y + h) / 2 + 100
        writenum(b)
        print(b)

    # show the image in a new window
    cv2.imshow("Security Feed", frame)

    # break the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        break

# stop the windows
cap.release()
cv2.destroyAllWindows()
