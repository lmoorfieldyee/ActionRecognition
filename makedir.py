"""
general purpose functions go in utils.py file
"""

import numpy as np
import os

def makedir(subject, description, action, number_of_video_samples):
    """
    Takes in subject name, action being performed, and descriptive title of how or where the action is being performed.
    Creates the directory structure to save the video frames. Does not return anything.

    :param subject: Name of the subject performing the action
    :param description: Description of how or where the subject is performing the action (i.e. sitting, standing,
    kithen, work, outside etc.).
    :param action: The action to be performed (wave, kiss, finger, salute, idle, heart)
    :param number_of_video_samples: How many empty video sample folders to create (default = 30)
    :return: Returns nothing. Creates directories to store video samples and their respective frames.
    """

    ROOT_PATH = './Raw_Data'
    # Assert that raw data file exists. If not tell the person to create manually
    # I'm not risking over-riding my data
    assert os.path.isdir(ROOT_PATH), "Ensure that Raw_Data folder is created in working directory"

    # Create subject folder path
    subject_data_path = os.path.join(ROOT_PATH, subject + "_" + description + "_Data")
    # Create action folder path
    action_video_path = os.path.join(subject_data_path, action)


    # check to see if above directories exist, if not then create them
    if not os.path.isdir(subject_data_path):
        os.makedirs(subject_data_path)
    if not os.path.isdir(action_video_path):
        os.makedirs(action_video_path)

    # set the number of video samples to collect (default is 30 for this project)
    num_vid_samples = np.arange(0, number_of_video_samples)

    # create 30 empty directories
    for vid_num in num_vid_samples:
        # create empty path for video folder
        vid_sample_path = os.path.join(action_video_path, str(vid_num))

        try:
            # create empty video folder
            os.makedirs(vid_sample_path)
        except:
            pass

if __name__ == "__main__":
    get_subject_name = input('enter subject name. >  ')
    get_location = input("Are you standing or sitting? > ")
    get_action = input('enter action to be recorded. > ')
    makedir(get_subject_name, get_location, get_action)
