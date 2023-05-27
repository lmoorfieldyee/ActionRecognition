import numpy as np
import cv2
from keypoints import Pipe
from makedir import makedir
import os


def overwrite_check(subject_path, ow):
    """
    :param subject_path: The file path to the current subject's video samples
    :param action: Specifies which action folder to look in to determine if videos already exist
    :param ow: Overwrite parameter - if set to yes, program will overwrite existing videos
    :return: If ow is set to 'n' then raise exception, else return 'y' to the overwrite param
    and program will overwrite existing video samples
    """

    # Check first video sample folder to see if it contains a video
    if (len(os.listdir(os.path.join(subject_path, str(0)))) != 0) and (ow == 'n'):

        # Print out directory at risk for being overwritten
        print('Directory Path: ', subject_path)

        # Get user input to overwrite or not
        ow = input('Danger! There are existing video samples in this folder, do you wish to '
                         'overwrite these videos?'
                         '\n(y/n) > ')
        # Assert that they provided a correct input. If not, stop the program (better safe than sorry)
        assert ((ow == 'y') or (ow == 'n')), print('incorrect input, stopping program')

        if ow == 'n':
            raise Exception("Overwrite Error. Check file path for existing videos.")
        else:
            return 'y'

def write_to_frame(frame_number, frames_per_vid, video_number, action, frame):
    """
    Takes in the current frame number, total frames in the video, and a frame to put the text on. Program checks to
    see if the frame is the last frame in the video sequence, if so then it provides a delay so subject can reset for
    next observation. Simply puts text on frame letting subject know which action and video sample they are recording
    for.
    :param frame_number: Current frame of video sample
    :param frames_per_vid: Total number of video samples to take (default 40)
    :param frame: Frame to put text on
    :return: The inputted frame with text on it
    """
    # If start of new sequence, then wait for 1 second
    if frame_number == frames_per_vid - 1:
        # Put some text on displayed frame
        cv2.putText(frame, "STARTING COLLECTION", (120, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 100), 3, cv2.LINE_AA)
        cv2.putText(frame, f'Collecting frames for {action} Video Number {video_number}', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # display opencv frame to screen
        cv2.imshow('OpenCV Frame', frame)
        cv2.waitKey(2000)
    else:
        cv2.putText(frame, f'Collecting frames for {action} Video Number {video_number}', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # display frame to screen
        cv2.imshow('OpenCV Frame', frame)

    return frame



def collect_raw_data(subject, description, action):
    """
    Takes in a subject and action to be performed. Creates a new directory for them if one does not exist. Records 30
    video samples of the action, with each video consisting of 40 frames, and saves each video's frames in a separate
    folder. Does not return anything.

    :param subject: Subject name performing the action
    :param description: Description of where or how the action is being performed (i.e. sitting, standing, home, outside
    etc.)
    :param action: The action being performed (i.e. wave, salute, finger, heart, kiss, idle)
    :return: Returns nothing. Saves all video samples to their respective folders
    """
    # defining the path to our raw data folder
    ROOT_PATH = './Raw_Data'

    # instantiate mediapipe object - only accessing attributes here
    model = Pipe()

    # create directories for new video samples
    # default number of video samples is 30
    makedir(subject, description, action, model.num_sample_videos)

    # create subject folder path
    subject_path = os.path.join(ROOT_PATH, subject + "_" + description + "_Data", action)

    # In order to avoid overwriting existing data, check to see if videos exist in current directory
    # Creating overwite flag - if set to yes('y') in below function, then existing videos will be overwritten
    overwrite = 'n'

    # overwrite function - if videos exists for subject, will check to see if you want to overwrite the existing videos
    overwrite = overwrite_check(subject_path, overwrite)

    # connect web camera
    cap = cv2.VideoCapture(0)

    # collect a video for each empty video folder in the subject's directory
    for vid_number in range(0, len(os.listdir(subject_path))):
        # create path to where we will save video frames - default is 40 frames per video
        for frame_num in range(0, model.frames_per_seq):
            path = os.path.join(subject_path, str(vid_number), str(frame_num))

            # read next frame
            ret, frame = cap.read()

            # convert frame to numpy array
            numpy_frame = np.array(frame)

            frame = write_to_frame(frame_num, model.frames_per_seq, vid_number, action, frame)

            # save landmarks to appropriate file
            np.save(path, numpy_frame)

            # break loop gracefully by pressing 'q'
            k = cv2.waitKey(10)
            if k == ord('q'):
                break

    # release video camera
    cap.release()
    # destroy all windows()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    get_subject_name = input('enter subject name. >  ')
    get_location = input("Are you standing or sitting? > ")
    get_action = input('enter action to be recorded. > ')
    collect_raw_data(get_subject_name, get_location, get_action)





##### THIS IS MY OLD CODE. ONLY KEEPING IT HERE FOR REFERENCE. #######
"""
path = './Raw_Data'
person = 'Dev_sitting' + '_Data'
data_folder = os.path.join(path, person)

actions = ['idle']
model = Pipe()

# connect web camera
cap = cv2.VideoCapture(0)

for action in actions:
    # loop through video files
    for vid_number in range(0, len(os.listdir(f'./{data_folder}/'+str(action)))):
        # create path to where we will save action sequences
        for frame_num in range(0, model.frames_per_seq):
            path = os.path.join(f'./{data_folder}', action, str(vid_number), str(frame_num))

            # read next frame
            ret, frame = cap.read()

            # If start of new sequence, then wait for 1 second
            if frame_num == model.frames_per_seq - 1:
                cv2.putText(frame, "STARTING COLLECTION", (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 100), 3, cv2.LINE_AA)
                cv2.putText(frame, f'Collecting frames for {action} Video Number {vid_number}', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # display frame to screen
                cv2.imshow('OpenCV Frame', frame)
                cv2.waitKey(2000)
            else:
                cv2.putText(frame, f'Collecting frames for {action} Video Number {vid_number}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # display frame to screen
                cv2.imshow('OpenCV Frame', frame)

                numpy_frame = np.array(frame)

            # save landmarks to appropriate file
            np.save(path, numpy_frame)

            # break loop gracefully by pressing 'q'
            k = cv2.waitKey(10)
            if k == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
"""