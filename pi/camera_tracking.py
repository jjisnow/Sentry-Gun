# camera_tracking.py
''' Show in camera window, the largest motion after various filters

Args:


'''

import argparse

import cv2 as cv


# import smbus

# bus = smbus.SMBus(1)
# address = 0x04


def writenum(value):
    # bus.write_byte(address, value)
    return -1


# disenabling opencl because it causes an error to do with the background subtraction
cv.ocl.setUseOpenCL(False)

# argument parser with minimum area for it to pick up as motion
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int,
                default=200,
                help="minimum area to pick up as motion")
args = vars(ap.parse_args())

# getting the video out from the webcam
cap = cv.VideoCapture(0)

# getting the background subtractor ready for use
fgbg = cv.createBackgroundSubtractorMOG2()

# setting state variables for color changes
is_gray = False
detect_mvt = False
detect_faces = False
detect_torso = False

# obtain classifiers for identifying objects
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')
torso_cascade = cv.CascadeClassifier('haarcascade_upperbody.xml')

# start the main loop which runs everything
while (1):
    # starting the loop while  reading from the video capture
    ret, frame = cap.read()

    # applying the bakground subtractor
    fgmask = fgbg.apply(frame)

    thresh = fgmask
    thresh = cv.GaussianBlur(thresh, (21, 21), 0)
    thresh = cv.threshold(thresh, 127, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=2)

    # making the image binary and adjusting it for the contouring
    (_, cnts, _) = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                                   cv.CHAIN_APPROX_SIMPLE)

    # Change the frame prior to drawing in contours
    if is_gray == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    if detect_mvt:
        # find the contours aroung the edges of the motion
        for c in cnts:
            if cv.contourArea(c) < args["min_area"]:
                continue

            # putting the contour area through the argument parser for maximum area
            c = max(cnts, key=cv.contourArea)

            # find the moments and centroid of the contour and draw in red for comparison
            M = cv.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

            # draw the actual contour found
            cv.drawContours(frame, c, -1, color=(255, 0, 0), thickness=2)

            # draw the rectangle around the object
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # draw a circle in the middle
            cv.circle(frame, (x + w // 2, y + h // 2), 2, (0, 255, 0), -1)

            a = (x + w) / 2
            writenum(a)
            print("width: {} ".format(a))

            b = (y + h) / 2 + 100
            writenum(b)
            print("height: {} ".format(b))

    if detect_faces:
        # Our operations on the frame (after contours drawn) come here
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        print('Faces found: ', len(faces))

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = frame[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # eyes are in the top 2/3 of faces, middle 60%
            eye_area = frame[
                         (y):(y + (h*2)//3),
                         (x + (w * 2) // 10):(x + (w * 8) // 10)]
            eyes = eye_cascade.detectMultiScale(eye_area, 1.1, 5)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(eye_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # smile in bottom 1/3 of faces, middle 60%
            smile_area = frame[
                         (y + (h * 2) // 3):(y + h),
                         (x + (w * 2) // 10):(x + (w * 8) // 10)]
            smiles = smile_cascade.detectMultiScale(smile_area, 1.1, 10)
            for (ex, ey, ew, eh) in smiles:
                cv.rectangle(smile_area, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    if detect_torso:
        # Our operations on the frame (after contours drawn) come here
        torsos = torso_cascade.detectMultiScale(frame, 1.3, 5)
        print('Torsos found: ', len(torsos))

        for (x, y, w, h) in torsos:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the image in a new window
    cv.imshow("Feed", frame)

    # OR show the mask
    # cv.imshow("Feed", thresh)

    # break the loop, wait 1ms whilst it checks for keypresses
    # 0xFF needed to filter only last 8 bits out
    key = cv.waitKey(1) & 0xFF

    # Save Image if 's' is pressed
    if key == ord("s"):
        out_file = "output.jpg"
        cv.imwrite(out_file, frame)
        print("{} saved!".format(out_file))

    elif key == ord("b"):
        # toggle black and white
        is_gray = not is_gray

    elif key == ord("m"):
        # toggle movement detection
        detect_mvt = not detect_mvt

    elif key == ord("f"):
        # toggle face detection
        detect_faces = not detect_faces

    elif key == ord("t"):
        # toggle torse detection
        detect_torso = not detect_torso

    elif key == ord("q"):
        break

# stop the windows
cap.release()
cv.destroyAllWindows()
