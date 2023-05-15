from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import os
import numpy as np


model = keras.models.load_model('./actions.h5')

# create class & data map
actions = os.listdir('./MP_Data')    # get names of classes
label_map = {label: num for num, label in enumerate(actions)}    # create mapping dict

# number of frames per video
frames_per_vid = 40

videos, labels = [], []
for action in actions:
    print(action)
    for vid_number in range(0, len(os.listdir('./MP_Data/'+str(action)))):
        window = []
        for frame_num in range(0, frames_per_vid):
            path = os.path.join('./MP_Data', action, str(vid_number), '{}.npy'.format(str(frame_num)))
            window.append(np.load(path))
        videos.append(window)
        labels.append(label_map[action])

X = np.array(videos)
y = to_categorical(labels).astype(int)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=50)

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


print(multilabel_confusion_matrix(ytrue, yhat))
print('accuracy soocre of {}'.format(accuracy_score(ytrue, yhat)))

print(ytrue)
print(yhat)