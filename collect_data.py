import numpy as np
import cv2
from keypoints import Pipe
import os

data_folder = 'Kieran_Data'

#actions = np.array(['wave', 'kiss', 'finger', 'salute', 'heart'])
actions = ['finger']
model = Pipe()

# connect web camera
cap = cv2.VideoCapture(0)

with model.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Set up data collection pipeline
    # loop through each action, collecting 30 videos per class, with each video being 50 frames long
    for action in actions:
        # loop through video files
        for vid_number in range(0, len(os.listdir(f'./{data_folder}/'+str(action)))):
            # create path to where we will save action sequences

            for frame_num in range(0, model.frames_per_seq):
                path = os.path.join(f'./{data_folder}', action, str(vid_number), str(frame_num))
                # read next frame
                ret, frame = cap.read()

                # make detections
                frame, results = model.pose_detection(frame, holistic)

                # draw landmarks on frame to be rendered
                model.draw_landmarks(frame, results)

                # If start of new sequence, then wait for 1 second
                if frame_num == model.frames_per_seq-1:
                    cv2.putText(frame, "STARTING COLLECTION", (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 100), 3, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting frames for {action} Video Number {vid_number}', (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # display frame to screen
                    cv2.imshow('OpenCV Frame', frame)
                    cv2.waitKey(3000)
                else:
                    cv2.putText(frame, f'Collecting frames for {action} Video Number {vid_number}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # display frame to screen
                    cv2.imshow('OpenCV Frame', frame)

                # extract pose, face, lh, & rh landmarks
                all_lm = model.extract_landmarks(results)

                # save landmarks to appropriate file
                np.save(path, all_lm)

                # break loop gracefully by pressing 'q'
                k = cv2.waitKey(10)
                if k == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
