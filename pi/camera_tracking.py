# camera_tracking.py
''' Show in camera window, the largest motion after various filters

Args:


'''

import argparse
import os
import numpy as np
import cv2 as cv


# import smbus

# bus = smbus.SMBus(1)
# address = 0x04


def writenum(value):
    # bus.write_byte(address, value)
    return -1


# Copied from opencv samples
bins = np.arange(256).reshape(256, 1)


def hist_curve(im):
    h = np.zeros((300, 256, 3))
    if len(im.shape) == 2:
        color = [(255, 255, 255)]
    elif im.shape[2] == 3:
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for ch, col in enumerate(color):
        hist_item = cv.calcHist([im], [ch], None, [256], [0, 256])
        cv.normalize(hist_item, hist_item, 0, 255, cv.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins, hist)))
        cv.polylines(h, [pts], False, col)
    y = np.flipud(h)
    return y


# disenabling opencl because it causes an error to do with the background subtraction
cv.ocl.setUseOpenCL(True)

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
is_blurry = False
show_threshold = False
show_rotation = False
show_histogram = False

# obtain classifiers for identifying objects
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')
torso_cascade = cv.CascadeClassifier('haarcascade_upperbody.xml')

# start the main loop which runs everything
while (1):
    # starting the loop while  reading from the video capture
    ret, frame = cap.read()

    # applying the background subtractor - for movement detection
    fgmask = fgbg.apply(frame)
    thresh = fgmask
    thresh = cv.GaussianBlur(thresh, (21, 21), 0)
    thresh = cv.threshold(thresh, 127, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=2)

    # making the image binary and adjusting it for the contouring
    (_, cnts, _) = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                                   cv.CHAIN_APPROX_SIMPLE)

    # Option for "Canny" Edge detection
    # frame = cv.Canny(frame, 50, 200, 3)

    # Change the frame prior to drawing in contours
    if is_gray == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    # Run detection algorithms as required
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
                       (y):(y + (h * 2) // 3),
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

    if is_blurry:
        kernel_tuple = (45, 45)

        # by using a filter kernel
        # kernel_tuple = (5, 5)
        # kernel = np.ones(kernel_tuple, np.float32) / 25
        # frame = cv.filter2D(frame, -1, kernel)

        # by using blur()
        # frame = cv.blur(frame, kernel_tuple)

        # by using GaussianBlur - must be + and odd
        frame = cv.GaussianBlur(frame, kernel_tuple, 0)

        # by using MedianBlur - removes salt and pepper noise
        # frame = cv.medianBlur(frame, 9)

        # by using Bilateral Filtering - keeps borders
        # frame = cv.bilateralFilter(frame, 9, 75, 75)

    if show_rotation:
        rows, cols, _ = frame.shape
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        frame = cv.warpAffine(frame, M, (cols, rows))

    # show the image in a new window
    # OR show the mask
    if show_threshold:
        thresh = cv.bitwise_and(frame, frame, mask=thresh)
        frame = thresh
        cv.imshow("Feed", frame)
    else:
        cv.imshow("Feed", frame)

    if show_histogram:
        curve = hist_curve(frame)
        x_offset = y_offset = 5
        frame[y_offset:y_offset + curve.shape[0],
            x_offset:x_offset + curve.shape[1]]\
            = curve
        cv.imshow("Feed", frame)

    # break the loop, wait 1ms whilst it checks for keypresses
    # 0xFF needed to filter only last 8 bits out
    key = cv.waitKey(1) & 0xFF

    # Save Image if 's' is pressed, in ascending file order
    if key == ord("s"):
        base_file = "output.jpg"

        base_file, file_extension = os.path.splitext(base_file)
        out_file = base_file

        i = 0
        while os.path.exists("{}{}".format(out_file, file_extension)):
            out_file = base_file + str(i)
            i += 1

        out_file += file_extension
        cv.imwrite(out_file, frame)
        print("{} saved!".format(out_file))

    # toggle black and white
    elif key == ord("b"):
        is_gray = not is_gray

    # toggle movement detection
    elif key == ord("m"):
        detect_mvt = not detect_mvt

    # toggle face detection
    elif key == ord("f"):
        detect_faces = not detect_faces

    # toggle torse detection
    elif key == ord("t"):
        detect_torso = not detect_torso

    # toggle blurriness
    elif key == ord("r"):
        is_blurry = not is_blurry

    # toggle threshold showing
    elif key == ord("h"):
        show_threshold = not show_threshold

    # toggle rotation showing
    elif key == ord("o"):
        show_rotation = not show_rotation

    # toggle histogram showing
    elif key == ord("i"):
        show_histogram = not show_histogram

    # quit
    elif key == ord("q"):
        break

# stop the windows
cap.release()
cv.destroyAllWindows()
