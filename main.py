import numpy as np
from tensorflow import keras
import cv2
from keypoints import Pipe
import os
from datetime import datetime
from datetime import timedelta
import socket

"""
NOTE: CHECK WHICH MODEL YOU ARE USING AS EACH MODEL HAS DIFFERENT FEATURES. THIS MEANS THAT YOU WILL NOT ONLY
NEED TO CHANGE THE MODEL BUT ALSO THE extract_landmark() FUNCTION TO PASS THROUGH CORRECT DATA. DEFAULT
MODEL IS clf_model3-837-0.09.hdf5. YOU ALSO NEED TO UPDATE THE NOSE LANDMARK DATA BEING PASSED TO UNITY AS IT IS IN 
A DIFFERENT SPOT FOR EACH DATASET.

BELOW IS A BREAKDOWN OF MODEL/EXTRACT_LANDMARK FCT'N PAIRS
1. actions.h5 -> extract_landmarks() fct'n
2. clf_model3-837-0.09.hdf5 (found in model3_loss folder) -> extract_landmarks3() fct'n
3. clf_model2-190-0.97.hdf5 (found in model2 folder) -> extract_landmarks4() fct'n

BELOW IS A BREAKDOWN OF MODEL/NOSE LANDMARK X,Y PAIRS
1. actions.h5 -> col index 0 (x); col index 1 (y)
2. clf_model2-190-0.97.hdf5 (found in model2 folder) -> col index 126 (x); col index 127 (y)
"""

# instantiate mediapipe
model = Pipe()

# load up action recognition neural network
clf_model = keras.models.load_model('clf_model2-190-0.97.hdf5')
# connect webcam
cap = cv2.VideoCapture(0)

# set up list to store rolling video frame landmarks
video_frames = []
# action performed tracker
detected_actions = [None]
# timestamp for each action
action_time = [datetime.now()]
# create time delta between each action (i.e. must wait 5 seconds before next action)
time_delta = timedelta(seconds=0)

# set threshold for action detection
threshold = 0.65
# get class names
actions = os.listdir('./Processed_Data')

# We need this program to talk with Unity
# Setting up socket to send data to
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Video tutorial used the ip "127.01.01.1" and 5052 port, but this does not exist on my system
serverAddressPort = ('127.0.0.1', 6942)

with model.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # read in next frame
        _, frame = cap.read()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


        # make mediapip detections
        frame, results = model.pose_detection(frame, model=holistic)

        # draw landmarks
        model.draw_landmarks(frame, results)

        # extract landmark values
        lh = model.extract_landmarks4(results)

        # initialize action being performed - default is no action
        action_being_performed = 'None'


        # append current frame's landmarks to video frame list
        video_frames.append(lh)

        # only keep the last 40 frames to do detections
        video_frames = video_frames[-model.frames_per_seq:]

        # if video_frames is more than 30 frames, check to see if action is performed
        if len(video_frames) == model.frames_per_seq:
            # make action predictions
            res = clf_model.predict(np.expand_dims(video_frames, axis=0))[0]

            # If action prediction confidence is larger than threshold, print the action
            if res[np.argmax(res)] > threshold:
                # check to see if action is idle
                if actions[np.argmax(res)] == 'idle':
                    detected_actions.append(actions[np.argmax(res)])
                    action_time.append(datetime.now())
                    video_frames = []

                # if not idle, then check to make sure same action is not performed two times in a row
                # and wait 1 second before making another action prediction (give user time to reset)
                else:
                    # Next action cannot be same as previous action
                    if actions[np.argmax(res)] != detected_actions[-1]:
                    #print(action_time[-1])
                    #print((action_time[-1] + time_delta))
                    # Must wait x (time_delta) seconds before next action
                    #if (datetime.now() - action_time[-1]) > time_delta:
                        detected_actions.append(actions[np.argmax(res)])
                        action_time.append(datetime.now())
                        print(actions[np.argmax(res)])
                        print(res[np.argmax(res)])
                        video_frames = []
                        cv2.putText(frame, f"{actions[np.argmax(res)]}", (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 100), 3, cv2.LINE_AA)

                        action_being_performed = actions[np.argmax(res)]

        # extract the nose landmark x & y values to update avatar head location
        x_nose_landmark = str(lh[126])
        y_nose_landmark = str(lh[127])

        # data to be sent to unity
        unity_data = (x_nose_landmark + "," + y_nose_landmark + "," + action_being_performed)

        # send nose location data and action data to unity
        sock.sendto(str.encode(unity_data), serverAddressPort)

        # show to frame
        cv2.imshow('Frame', frame)

        # press k to break loop gracefully
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
