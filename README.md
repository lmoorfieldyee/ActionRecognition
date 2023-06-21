# ActionRecognition

### Research Aim:

Action recognition is the process of interpreting human motion using computer vision. Use cases for the technology is broad with applications across, but not limited to, security systems, human-computer interactions, human behavior analysis, and entertainment and gaming. 

With the rise of large language models, this project focuses on creating a deep learning model to identify human gestures to further develop human-computer interactions. As a large portion of communication is non-verbal, having a model to identify a person's non-verbal communication is important to develop systems which can fully assess all aspects of an interaction before responding. This will allow for more personalised and precise communication between AI models and humans. This project was inspired by Nicholas Renotte's sign language detection project, https://www.youtube.com/watch?v=doDUihpj6ro&t=5048s&ab_channel=NicholasRenotte, and much of the project builds off the code he uses.

The select actions were chosen due to their distinct and clear motions, and unambiguous meanings. These actions provided a strong baseline model, and framework, where more subtle actions can easily be added to in the future. The six actions which can be detected are:

1) Wave
2) Kiss
3) Heart
4) Finger
5) Salute 
6) Idle

#### Project Methodology:
The project is divided into 5 parts; data collection, data pre-processing, exploratory data analysis, model training and evaluation, and model deployment. The project is structured in such a way that anyone can collect their own data, create their own model, and deploy it in a simple real time application. Below is a step by step guide to creating your own model, or replicate mine. If you want to replicate my results then please download my processed data, which can be found here, and skip to step #.

**Steps**
1. Raw Data Collection: To collect your own training video samples, start by running the "collect_raw_video.py" script. This will prompt you to enter the name of the subject performing the action, describe where the subject is performing the action (standing, sitting, etc.), and which action is being performed. The script will then do the following: 
i. Create the file structure to house new data. An example of the raw data directory structure is below, and please be very careful with your naming conventions as it is easy to overwrite existing data.

<p style="text-align: center;">

- Raw_Data
    - Liam_standing (subject_name + description)
        - Heart (action being performed)
            - Video_sample_0
                - video_frame_0
                - video_frame_1
                - ...
                - video_frame_39
            - ...
            - Video_sample_29
        - Salute (action being performed)
            - Video_sample_0
            - ...
            - Video_sample_29
        - ...
        - Wave
    - Liam_sitting 
        - Heart (action being performed)
        - ...
        - Wave
    - Etc.
</p>
            
ii. Proceed to collect 30 video samples of the action to be performed (A video capture will display to display the start and end of each video sample)
3) save each video as a series of numpy arrays (each video has 40 frames/numpy arrays). You will need to run this script multiple times to gather all the data you require for your different actions.

The model was trained on 1260 videos (210 videos per class) and each video was generated and collected individually. The project covers the full machine learning pipeline including scripts for data collection, pre-processing, feature engineering, model training, and model deployment. 
2.  

