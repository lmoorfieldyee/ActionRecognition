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
A DIFFERENT SPOT FOR EACH DATASET. A YOU ALSO NEED TO UPDATE THE HANDS_ABOVE_ELBOW() FUNCTION... REALLY NEED TO
JUST AUTOMATE THIS EUGH.

BELOW IS A BREAKDOWN OF MODEL/EXTRACT_LANDMARK FCT'N PAIRS
1. actions.h5 -> extract_landmarks() fct'n
2. clf_model3-837-0.09.hdf5 (found in model3_loss folder) -> extract_landmarks3() fct'n
3. clf_model2-190-0.97.hdf5 (found in model2 folder) -> extract_landmarks4() fct'n

BELOW IS A BREAKDOWN OF MODEL/NOSE LANDMARK X,Y PAIRS
1. actions.h5 -> col index 0 (x); col index 1 (y)
2. clf_model2-190-0.97.hdf5 (found in model2 folder) -> col index 126 (x); col index 127 (y)

BELOW IS A BREAKDOWN OF MODEL MAPPING FOR HANDS_ABOVE_ELBOW()
1. actions.h5 -> MODEL NUMBER = 1 (DEFAULT)
2. clf_model2-190-0.97.hdf5 (found in model2 folder) -> MODEL NUMBER = 2
"""

def hand_above_elbow(landmark_list, model_number=1):
    """
    Takes in processed frame landmarks and deep learning model. Calculates the distance between wrist landmarks and
    their respective elbows. If a wrist is above an elbow then returns True as an action could be taking place.
    Other-wise returns False.

    :param results: Takes in mediapipe landmark results from processed frame
    :param model_number: Takes either 1 or 2 which specifies which model you are using.
    1 for model 'actions.h5' (default) and 2 for 'clf_model2-190-0.97.hdf5'
    :return: True or False
    """
    if model_number == 2:
        # get elbow landmarks (13 (52 (178) ,53 (179)) & 14 (56 (182) , 57 (183)))
        left_elbow_y = landmark_list[179]
        right_elbow_y = landmark_list[183]

        # get wrist landmarks (0,1 & 63,64)
        left_wrist_y = landmark_list[1]
        right_wrist_y = landmark_list[64]
    else:
        # get elbow landmarks (13 (52 (178) ,53 (179)) & 14 (56 (182) , 57 (183)))
        left_elbow_y = landmark_list[52]
        right_elbow_y = landmark_list[56]

        # get wrist landmarks (0,1 & 63,64)
        left_wrist_y = landmark_list[1537]
        right_wrist_y = landmark_list[1600]

    if ((left_wrist_y != 0) & (left_wrist_y < left_elbow_y)) | ((right_wrist_y != 0) & (right_wrist_y < right_elbow_y)):
        return True
    else:
        return False

def load_model(model_number=1):
    if model_number==2:
        return keras.models.load_model('./model2/clf_model2-190-0.97.hdf5')
    else:
        return keras.models.load_model('actions.h5')

def send_unity_data(landmark_list, action_being_performed, data_port, model_number=1):
    if model_number==2:
        # extract the nose landmark x & y values to update avatar head location
        x_nose_landmark = str(landmark_list[126])
        y_nose_landmark = str(landmark_list[127])

        # data to be sent to unity
        unity_data = (x_nose_landmark + "," + y_nose_landmark + "," + action_being_performed)

        # send nose location data and action data to unity
        sock.sendto(str.encode(unity_data), data_port)
    else:
        # extract the nose landmark x & y values to update avatar head location
        x_nose_landmark = str(landmark_list[0])
        y_nose_landmark = str(landmark_list[1])

        # data to be sent to unity
        unity_data = (x_nose_landmark + "," + y_nose_landmark + "," + action_being_performed)

        # send nose location data and action data to unity
        sock.sendto(str.encode(unity_data), data_port)

def percent_action_complete(video_len, frames_per_action, frame_to_write):
    percent_complete = video_len / frames_per_action
    x_dist = 120 + int((520-120)*percent_complete)
    cv2.rectangle(frame_to_write, (120, 400), (x_dist, 430), color=(155,100,50), thickness=-1)
    cv2.putText(frame_to_write, f"Action Complete Percent: {percent_complete}", (200, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def reset_video_list(video_frame_list, append_time_list, time_delay):
    # We need to add a time delta as it's quite common for the model to lose track of the hand for a second

    # Check to make sure that frame list is being added to, if not just return lists
    if len(append_time_list) == 0:
        return video_frame_list, append_time_list
    # If actively being added to, if it's been longer than 1 second since last frame reset lists
    else:
        if (datetime.now() - append_time_list[-1]) > time_delay:
            # reset video_frames list to empty
            video_frame_list, append_time_list = [], []
            return video_frame_list, append_time_list
        else:
            return video_frame_list, append_time_list



# instantiate mediapipe
model = Pipe()

# set which model you're using
model_number = 2

# load up action recognition neural network
clf_model = load_model(model_number)

# connect webcam
cap = cv2.VideoCapture(0)

# set up list to store rolling video frame landmarks
video_frames = []

# set up list to store time of last video frame appended
video_frame_append_time = []

# action performed tracker
detected_actions = [None]

# timestamp for each action
action_time = [datetime.now()]

# create time delta between each action (i.e. must wait 5 seconds before next action)
time_delta = timedelta(seconds=1)

# set threshold for action detection
threshold = 0.6

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
        model.draw_landmarks2(frame, results)

        # extract landmark values
        lh = model.extract_landmarks4(results)

        # initialize action being performed - default is no action
        action_being_performed = 'None'

        # draw action percent complete bar
        cv2.rectangle(frame, (120, 400), (520, 430), color=(155, 255, 0), thickness=2)
        if len(video_frames) == 0:
            cv2.putText(frame, f"Action Complete Percent: 0", (200, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Only start appending video frames if hands are above elbows. If hands are below elbows then we are
        # assuming that no action is being performed.
        if hand_above_elbow(lh, model_number):

            # append current frame's landmarks to video frame list & update time
            video_frames.append(lh)
            video_frame_append_time.append(datetime.now())

            percent_action_complete(len(video_frames), model.frames_per_seq, frame)

            # only keep the last 40 frames to do detections
            video_frames = video_frames[-model.frames_per_seq:]

            # if video_frames is more than 40 frames, check to see if action is performed
            if len(video_frames) == model.frames_per_seq:

                print('Predicting Action!!!')

                # make action predictions
                res = clf_model.predict(np.expand_dims(video_frames, axis=0))[0]

                # If action prediction confidence is larger than threshold, print the action
                if res[np.argmax(res)] > threshold:
                    # check to see if action is idle
                    if actions[np.argmax(res)] == 'idle':
                        detected_actions.append(actions[np.argmax(res)])
                        action_time.append(datetime.now())

                    # if not idle, then check to make sure same action is not performed two times in a row
                    # and wait 1 second before making another action prediction (give user time to reset)
                    else:
                        # append action to detected actioon list
                        detected_actions.append(actions[np.argmax(res)])
                        action_time.append(datetime.now())
                        print(actions[np.argmax(res)])
                        print(res[np.argmax(res)])
                        # reset frame list for next action to stop rapid fire predictions
                        video_frames = []
                        cv2.putText(frame, f"{actions[np.argmax(res)]}", (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 100), 3, cv2.LINE_AA)

                        action_being_performed = actions[np.argmax(res)]

                        # reset video frame after a prediction has been made
                        video_frames = []
                        video_frame_append_time = []

        # If hands are below elbows then we want to keep the video_frames for the next detection empty
        else:
            video_frames, video_frame_append_time = reset_video_list(video_frames, video_frame_append_time, time_delta)

        # send data to given serverAddressPort
        send_unity_data(lh, action_being_performed, serverAddressPort, model_number)

        # show to frame
        cv2.imshow('Frame', frame)

        # press k to break loop gracefully
        k = cv2.waitKey(10)
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
