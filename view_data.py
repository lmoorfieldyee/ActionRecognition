import numpy as np
import cv2
from keypoints import Pipe
import os

path = './Raw_Data_Testing_Folder'
person = 'Kieran_sitting'
data_folder = os.path.join(path, person)

actions = ['wave']
model = Pipe()

path = "./Raw_Data/Pierce_sitting_dining_room_Data/wave"
#with model.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
for video_num in range(30):
    for frame_num in range(40):
        numpy_path = os.path.join(path, str(video_num), str(frame_num) + ".npy")
        print(numpy_path)
        frame = np.load(numpy_path)
        # make mediapipe detections
        #frame, results = model.pose_detection(frame, holistic)
        # draw landmarks on frame to be rendered
        #model.draw_landmarks(frame, results)
        cv2.imshow('frame', frame)
        cv2.waitKey(15)

cv2.destroyAllWindows()
