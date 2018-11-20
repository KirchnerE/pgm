import numpy as np
import scipy.io as sio
from sklearn import svm

# get eeg out of file:
s01 = sio.loadmat("data/s01.mat")
eeg   = s01["eeg"]

srate  = eeg["srate"][0,0][0,0]
rest_data   = eeg["rest"][0,0]
noise_data  = eeg["noise"][0,0]
left_data   = eeg["movement_left"][0,0]
right_data  = eeg["movement_right"][0,0]
events_data = eeg["movement_event"][0,0]


def downsample(data):
    N = data.shape[1]
    i = np.arange(0, N, 4)
    return data[:, i]


event_indices = np.where(events_data[0,:] == 1)[0]
left_vectors = []
right_vectors = []
for i in event_indices:
    # get the half second before event
    downsampled_left = downsample(left_data[:,i-int(0.5*srate):i])
    left_vectors += [downsampled_left.reshape(-1)]
    downsampled_right = downsample(right_data[:,i-int(0.5*srate):i])
    right_vectors += [downsampled_right.reshape(-1)]

left_matrix = np.array(left_vectors)
right_matrix = np.array(right_vectors)
X = np.append(left_matrix, right_matrix, 0)
# left movement: 0; right movement: 1
left_labels = np.zeros(left_matrix.shape[0])
right_labels = np.ones(right_matrix.shape[0])
y = np.append(left_labels, right_labels, 0)

# create support vector machine:
clf = svm.SVC(gamma=0.05) # TODO: find good gamma on validation set
clf.fit(X, y)

# Test on training data -> TODO: more data with train and validation set
pred_l = clf.predict(left_matrix[0,:].reshape(1,-1))
pred_r = clf.predict(right_matrix[0,:].reshape(1,-1))
print(pred_l)
print(pred_r)
