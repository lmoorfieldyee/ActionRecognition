import sys
# Since notebooks are saved in subdirectory, we need to add root project directory to python search paths to access
# other parts of program
sys.path.append("../") # go to parent dir

import numpy as np
from functions.keypoints import Pipe
import os

"""
Be careful with this script as it will overwrite any old processed data you may have. It is not the worst thing
as it does not overwrite the underlying raw data so you can always re-run the processing, but it can be quite time 
consuming.
"""

# instantiate mediapipe model
model = Pipe()

# set directory for raw data and directory for data post processing
root_raw = '../Raw_Data'
root_processed = '../Processed_Data'

raw_data_files = os.listdir(root_raw)
actions = ['wave', 'salute', 'kiss', 'idle', 'heart', 'finger']

# set up holistic model
with model.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for file in raw_data_files:
        print(file)
        for action in actions:
            print(action)
            file_path = os.path.join(root_processed, action, file)
            if os.path.isdir(file_path)==False:
                os.makedirs(file_path)
            for video_num in range(model.num_sample_videos):
                file_path = os.path.join(root_processed, action, file, str(video_num))
                if os.path.isdir(file_path)==False:
                    os.makedirs(file_path)
                for frame_num in range(model.frames_per_seq):

                    # create the path to the raw video frame
                    frame_path = os.path.join(root_raw, file, action, str(video_num), str(frame_num) + '.npy')

                    # load the numpy frame
                    frame = np.load(frame_path)

                    # make mediapipe detections
                    frame, results = model.pose_detection(frame, holistic)

                    # draw landmarks on frame to be rendered
                    #model.draw_landmarks(frame, results)

                    #cv2.imshow('frame', frame)

                    # extract pose, face, lh, & rh landmarks
                    all_lm = model.extract_landmarks(results)

                    # define the output path to save the numpy files to
                    # should be "Processed_Data/action_name/video_sample_number/frame_num.npy"
                    output_path = os.path.join(file_path, file + str(frame_num) + ".npy")

                    # save new file
                    np.save(output_path, all_lm)
