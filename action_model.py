from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score
import keras
import os
import numpy as np
from keypoints import Pipe

# create class & data map
root_data = './Processed_Data'
actions = os.listdir(root_data)    # get names of classes
label_map = {label: num for num, label in enumerate(actions)}    # create mapping dict
print(label_map)
print(actions)

# Instantiate model
model = Pipe()

# number of frames per video
frames_per_vid = model.frames_per_seq

videos, labels = [], []

for action in actions:
    print(action)
    # get names of all names of all people who provided data
    diff_data_subjects = os.listdir(os.path.join(root_data, action))
    for data_subject in diff_data_subjects:
        for vid_number in range(0, model.num_sample_videos):
            window = []
            for frame_num in range(0, frames_per_vid):
                path = os.path.join(root_data, action, data_subject, str(vid_number),
                                    data_subject+'{}.npy'.format(str(frame_num)))
                window.append(np.load(path))
            videos.append(window)
            labels.append(label_map[action])


X = np.array(videos)
y = to_categorical(labels).astype(int)
print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=50)

print(X_train.shape)
print(y_train.shape)
'''
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

clf_model = Sequential()
clf_model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(frames_per_vid, 1662)))
clf_model.add(LSTM(128, return_sequences=True, activation='relu'))
clf_model.add(LSTM(64, return_sequences=False, activation='relu'))
clf_model.add(Dense(64, activation='relu'))
clf_model.add(Dense(32, activation='relu'))
clf_model.add(Dense(6, activation='softmax'))

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
clf_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['categorical_accuracy'])
clf_model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])

yhat = clf_model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


print(confusion_matrix(ytrue, yhat))
print('accuracy soocre of {}'.format(accuracy_score(ytrue, yhat)))


clf_model.save('actions.h5')
del clf_model'''