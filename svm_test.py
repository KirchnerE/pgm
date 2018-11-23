import numpy as np
import mne
from sklearn import svm


epochs_train = mne.read_epochs('preprocessed_files/pp_S001R07-epo.fif')
epochs_test = mne.read_epochs('preprocessed_files/pp_S001R07-epo.fif')

def compute_X_y(epochs):
    # Assume left is 1, right is two. Maybe this has to be swapped
    left_events = epochs['1']
    right_events = epochs['2']

    #first dimension events, second dimension features (channels*time)
    left_matrix = left_events.get_data()
    num_left_events = left_matrix.shape[0]
    left_matrix = left_matrix.reshape(num_left_events, -1)
    right_matrix = right_events.get_data()
    num_right_events = right_matrix.shape[0]
    right_matrix = right_matrix.reshape(num_right_events, -1)

    left_labels = np.ones((num_left_events, 1))
    right_labels = 2 * np.ones((num_right_events, 1))

    X = np.append(left_matrix, right_matrix, 0)
    y = np.append(left_labels, right_labels, 0)
    return X, y

X_train, y_train = compute_X_y(epochs_train)
X_test, y_test = compute_X_y(epochs_test)

# create support vector machine:
clf = svm.SVC(gamma=0.005) # TODO: find good gamma on validation set
clf.fit(X_train, y_train)

# Test on training data -> TODO: more data with train and validation set
pred = clf.predict(X_test)

num_test_events = y_test.shape[0]

print("\n\n")
print(pred)
print(y_test.T)
correctly_classified = 1 - np.abs(pred-y_test.T).sum()/num_test_events
print(correctly_classified, "% of the samples were correctly classified")
