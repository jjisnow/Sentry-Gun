# camera_tracking.py
''' Show in camera window, the largest motion after various filters

Args:


'''

import argparse

import cv2


# import smbus

# bus = smbus.SMBus(1)
# address = 0x04


def writenum(value):
    # bus.write_byte(address, value)
    return -1


# disenabling opencl because it causes an error to do with the background subtraction
cv2.ocl.setUseOpenCL(False)

# argument parser with minimum area for it to pick up as motion
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int,
                default=200,
                help="minimum area to pick up as motion")
args = vars(ap.parse_args())

# getting the video out from the webcam
cap = cv2.VideoCapture(0)

# getting the background subtractor ready for use
fgbg = cv2.createBackgroundSubtractorMOG2()

# setting state variables for color changes
is_gray = False

# start the main loop which runs everything
while (1):
    # starting the loop while  reading from the video capture
    ret, frame = cap.read()

    # applying the bakground subtractor
    fgmask = fgbg.apply(frame)

    thresh = fgmask
    thresh = cv2.GaussianBlur(thresh, (21, 21), 0)
    thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # making the image binary and adjusting it for the contouring
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Change the frame prior to drawing in contours
    if is_gray == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # find the contours aroung the edges of the motion
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # putting the contour area through the argument parser for maximum area
        c = max(cnts, key=cv2.contourArea)

        # find the moments and centroid of the contour and draw in red for comparison
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

        # draw the actual contour found
        cv2.drawContours(frame, c, -1, color=(255, 0, 0), thickness=2)

        # draw the rectangle around the object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # draw a circle in the middle
        cv2.circle(frame, (x + w // 2, y + h // 2), 2, (0, 255, 0), -1)

        a = (x + w) / 2
        writenum(a)
        print("width: {} ".format(a))

        b = (y + h) / 2 + 100
        writenum(b)
        print("height: {} ".format(b))

    # Our operations on the frame (after contours drawn) come here

    # show the image in a new window
    cv2.imshow("Feed", frame)

    # OR show the mask
    # cv2.imshow("Feed", thresh)

    # break the loop, wait 1ms whilst it checks for keypresses
    # 0xFF needed to filter only last 8 bits out
    key = cv2.waitKey(1) & 0xFF

    # Save Image if 's' is pressed
    if key == ord("s"):
        out_file = "output.jpg"
        cv2.imwrite(out_file, frame)
        print("{} saved!".format(out_file))

    elif key == ord("b"):
        # toggle black and white
        is_gray = not is_gray

    elif key == ord("q"):
        break

# stop the windows
cap.release()
cv2.destroyAllWindows()
