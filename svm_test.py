import numpy as np
import mne
from sklearn import svm


epochs_1 = mne.read_epochs('preprocessed_files/pp_S001R03-epo.fif')
epochs_2 = mne.read_epochs('preprocessed_files/pp_S001R07-epo.fif')
epochs_3 = mne.read_epochs('preprocessed_files/pp_S001R11-epo.fif')

def compute_X_y(epochs):
    # Assume left is 1, right is two. Maybe this has to be swapped
    left_events = epochs['1']
    right_events = epochs['2']

    #first dimension events, second dimension features (channels*time)
    left_matrix = left_events.get_data()
    left_matrix = left_matrix.swapaxes(0, 1)
    left_matrix = left_matrix.reshape(-1, 8) 

    num_left_events = left_matrix.shape[0]

    right_matrix = right_events.get_data()
    right_matrix = right_matrix.swapaxes(0, 1)
    right_matrix = right_matrix.reshape(-1, 8) 

    num_right_events = right_matrix.shape[0]
    #right_matrix = right_matrix.reshape(num_right_events, -1)

    left_labels = np.ones((num_left_events, 1))
    right_labels = 2 * np.ones((num_right_events, 1))


    X = np.concatenate((left_matrix, right_matrix), axis=0)
    y = np.concatenate((left_labels, right_labels), axis=0)
    return X, y

#X1, y1 = compute_X_y(epochs_1)
X, y = compute_X_y(epochs_2)
#X3, y3 = compute_X_y(epochs_3)
'''
X = np.concatenate((X1, X2, X3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices,:]
y = y[indices,:]

num_samples = X.shape[0]
X_train = X[:int(np.floor(0.75*num_samples)),:]
X_test = X[int(np.floor(0.75*num_samples)):,:]
y_train = y[:int(np.floor(0.75*num_samples)),:]
y_test = y[int(np.floor(0.75*num_samples)):,:]

'''

# create support vector machine:
clf = svm.SVC(kernel="linear", gamma="auto") # TODO: find good gamma on validation set
clf.fit(X, y)

# Test on training data -> TODO: more data with train and validation set
pred = clf.predict(X)

num_test_events = y.shape[0]

print("\n\n")
print(pred)
print(y.T)
correctly_classified = 1 - np.abs(pred-y.T).sum() / num_test_events
print(correctly_classified, "% of the samples were correctly classified")
