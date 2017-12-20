# #############################################################################
# PEER testing for fMRI data

# Screen size: 1680 x 1050 - from follow_the_dot_lastrun.py in ~/Desktop/HBN_Peer

import os
import sys
import numpy as np
import nibabel as nib
from nilearn import image
from sklearn.svm import SVR
from nilearn import plotting
import matplotlib.pyplot as plt

# DON'T FORGET MOTION CORRECTION BEFORE IMPORTING DATA

# Import data
img = nib.load('peer2_processed.nii.gz')
data = img.get_data()
print(img.shape)

# For visualization
for num in [0, 50]:
    visual = image.index_img(img, num)

    slice_data = visual.get_data()[:, :, 15:30]

    plotting.plot_stat_map(visual)
    plotting.show()

# Data processing (flatten voxel intensity from 2D to 1D)
# Use RAVEL function python (np.ravel())


# Train SVR classifier on first two calibrations and predict values for third calibration
train_vectors = 'Contains vectors of voxel intensity values for each axial slice'
test_vectors = 'Contains vectors of voxel intensity values for each axial slice'
train_x = 'Contains x coordinates of first two calibration fixations'
train_y = 'Contains y coordinates of first two calibration fixations'
test_x = 'Contains x coordinates of third calibration fixations'
test_y = 'Contains x coordinates of third calibration fixations'

classifier = SVR(kernel='poly', degree=10, C=100, epsilon=.001)
predicted_x = classifier.fit(train_vectors, train_x).predict(test_x)
predicted_y = classifier.fit(train_vectors, train_y).predict(test_y)

# Plot SVR predictions against targets
plt.figure()
plt.plot(predicted_x, predicted_y, color='b')
plt.plot(train_x, train_y, color='r.')
plt.xlabel('x-position')
plt.ylabel('y-position')
plt.title('Support Vector Regression - PEER')
plt.legend()
plt.show()



