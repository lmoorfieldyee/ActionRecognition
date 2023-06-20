# ActionRecognition
This computer vision application uses a LSTM neural network to detect the following six unique actions: 
1) wave
2) kiss
3) heart 
4) finger
5) salute 
6) idle 

The model was trained on 1260 videos (210 videos per class) and each video was generated and collected individually. The project covers the full machine learning pipeline including scripts for data collection, pre-processing, feature engineering, model training, and model deployment. 

This project was inspired by Nicholas Renotte's sign language detection project, https://www.youtube.com/watch?v=doDUihpj6ro&t=5048s&ab_channel=NicholasRenotte, and much of the project builds off the code he uses.

### Overview
Project is divided into 5 parts; data collection, data pre-processing, model training and evaluation, and model deployment. The project is structured in such a way that anyone can collect their own data, create their own model, and deploy it in a simple real time application.

In to do this, you must follow the below steps.

1) Collect raw data (or use pre-processed data provided below) by running the "collect_raw_video.py" script. This will do the following: 1) create the file structure to house the newly generated data; 2) collect 30 video samples of the action to be performed; 3) save each video as a series of numpy arrays (each video has 40 frames/numpy arrays). You will need to run this script multiple times to gather all the data you require for your different actions.
2) 

