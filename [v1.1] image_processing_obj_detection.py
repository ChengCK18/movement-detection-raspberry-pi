# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
import requests
from datetime import datetime


# Discord header and url
# How to get the discord authorization token and desired discord channel URL
# https://www.youtube.com/watch?v=DArlLAq56Mo&t=6s
# Thanks CodeDict :)
header = {
    'authorization': "The Discord Authentication Token of this device's account"
}
url = 'Your desired discord channel for the alert to be sent to. Could be your own discord or server et cetera...'

# Video capture 0 for webcam
cap = cv2.VideoCapture(0)


# Initialization of variables
base_img = []
erode_kernel = np.ones((5, 5), np.uint8)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int((1 / int(fps)) * 780)
reset_delay = 5
record = False
in_session = False

# Define the codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# Loop to actively take new frame and process it
while True:
    _, img = cap.read()  # Grab new frame from webcam
    # Flip frame vertically, webcam was positioned upside down
    img = cv2.flip(img, -1)
    img = cv2.flip(img, 1)  # Flip frame horizontally, mirror effect

    # convert frame from RGB to single gray channel and apply gaussian blur to smoothen image (Reduce noise)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # binarize the image to 0/1 based on the threshold value determined using Otsu Thresholding method
    thresh, gray_img = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Set frame as background/base frame if base img is not defined yet
    if len(base_img) == 0:
        base_img = gray_img

    # Get the abs diff between the current frame and base frame
    abs_img = cv2.absdiff(base_img, gray_img)
    # Perform erosion to eliminate random noise
    abs_img = cv2.erode(abs_img, erode_kernel, iterations=3)

    # To provide boundary for connected blobs and draw bounding rect of those boundary on the original frame
    (contours, _) = cv2.findContours(
        abs_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # If blobs area is more than x value, means movement/foreign object detected
        if cv2.contourArea(contour) > 800:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print("Found um...movement?")
            record = True  # To initiate recording session with webcam
    if record:
        if(not in_session):  # To initialize vairables for the start of recording
            # Name of file reflects the current date and time
            current_time = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
            file_name = current_time + ".avi"
            # define videowriter object to write sequences of frame into a file
            out = cv2.VideoWriter(file_name, fourcc, 20.0, (640, 480))

            # Recording will run for approximately 30
            record_end_time = time.time() + 14  # run for 30 seconds
            print("Recording...")
            in_session = True

        if time.time() < record_end_time:  # continue to record by adding the current frame to the file, if time ends stop writing
            out.write(img)  # write new frame into the defined file
        else:
            print("Recording complete.")
            # Send DM to discord as an alert
            # Payload is the message while files is the recorded video
            payload = {
                'content': "Found um...movement?\nDate:" + str(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
            }
            files = {'file': open(file_name, 'rb')}
            # send a post request to the discord channel url
            # In this case, this device is signed on to a separate discord account to send direct message to my own discord account
            r = requests.post(url, data=payload, headers=header, files=files)

            out.release()  # to free allocated resources and security purpose
            # to reset base image (To prevent device from constantly detecting unexpected noises)
            base_img = []
            record = False  # reset record flag
            in_session = False

    # cv2.imshow('Display',abs_img) #Show every frame output for debugging purpose
    key = cv2.waitKey(delay)

    if key == 27:  # esc key to exit
        cv2.destroyAllWindows()
        break
    if key == ord('a'):
        print("Resetting background image in (%d) seconds..." % (reset_delay))
        time.sleep(reset_delay)
        # To flush the videocapture buffer to prevent it from using frames from buffer as base frame instead of new frame after delay
        cap = None
        cap = cv2.VideoCapture(0)
        base_img = []
        print("Reset complete")

cap.release()  # to free allocated resources
