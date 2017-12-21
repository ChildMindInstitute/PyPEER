# #############################################################################
# PEER testing for fMRI data

# Screen size: 1680 x 1050 - from follow_the_dot_lastrun.py in ~/Desktop/HBN_Peer

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from sklearn.svm import SVR
from nilearn import plotting
import matplotlib.pyplot as plt

# DON'T FORGET MOTION CORRECTION BEFORE IMPORTING DATA

monitor_width = 1680
monitor_height = 1050

# Import data
img = nib.load('peer2_processed.nii.gz')
data = img.get_data()
testing = nib.load('peer1_processed.nii.gz')
print(img.shape)

# Vectorize data into single np array

listed = []

for tr in range(int(data.shape[3])):

    tr_data = data[:, :, 15:30, tr]
    vectorized = np.array(tr_data.ravel())
    listed.append(vectorized)

df = np.asarray(listed)

# Create np array that contains all fixation locations, separated by x and y coordinates

fixations = pd.read_csv('stim_vals.csv')
x_targets = np.repeat(np.array(fixations['pos_x']), 5)*monitor_width/2
y_targets = np.repeat(np.array(fixations['pos_y']), 5)*monitor_height/2

# # For visualization
# for num in [0, 50]:
#     visual = image.index_img(img, num)
#     slice_data = visual.get_data()[:, :, 15:30]
#     plotting.plot_stat_map(visual)
#     plotting.show()

# Train SVR classifier on first two calibrations and predict values for third calibration
train_vectors = 'Contains vectors of voxel intensity values for each axial slice'
test_vectors = 'Contains vectors of voxel intensity values for each axial slice'
test_vectors = 'Contains x coordinates of third calibration fixations'
test_vectors = 'Contains x coordinates of third calibration fixations'

# Testing - Remove after validation
train_vectors = df
train_x = x_targets

classifier = SVR(kernel='rbf', degree=25, C=1000, epsilon=.001)
# Validation
validation_x = classifier.fit(train_vectors, x_targets).predict(train_vectors)
validation_y = classifier.fit(train_vectors, y_targets).predict(train_vectors)
# Testing
predicted_x = classifier.fit(train_vectors, x_targets).predict(test_vectors)
predicted_y = classifier.fit(train_vectors, y_targets).predict(test_vectors)

# Plot SVR predictions against targets
plt.figure()
plt.plot(predicted_x, predicted_y, color='b')
plt.plot(x_targets, y_targets, color='r.')
plt.xlabel('x-position')
plt.ylabel('y-position')
plt.title('Support Vector Regression - PEER')
plt.legend()
plt.show()



