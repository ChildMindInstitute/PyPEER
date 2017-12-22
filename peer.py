# #############################################################################
# PEER testing for fMRI data

# Screen size: 1680 x 1050 - from follow_the_dot_lastrun.py in ~/Desktop/HBN_Peer

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from sklearn.svm import SVR
from nilearn import plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

monitor_width = 1680
monitor_height = 1050

# Import data

img = nib.load('PEER1_resampled.nii.gz')
data = img.get_data()
testing = nib.load('PEER2_resampled.nii.gz')
testing_data = testing.get_data()

# Vectorize data into single np array

listed = []
listed_testing = []

for tr in range(int(data.shape[3])): # training with 5TRs from each fixation (averaged or individual)
# for tr in [i*5 for i in range(27)]: # training with one TR from each fixation

    tr_data = data[:, :, 10:18, tr]
    te_data = data[:, :, 10:18, tr]

    # tr_data = data[:, :, 18:25, tr]
    # tr_data = data[:, :, 15:30, tr]
    # te_data = testing_data[:, :, 18:25, tr]
    # te_data = testing_data[:, :, 15:30, tr]
    vectorized = np.array(tr_data.ravel())
    vectorized_testing = np.array(te_data.ravel())
    listed.append(vectorized)
    listed_testing.append(vectorized_testing)

# CHECK AFNI, FSL, NILEARN FOR 3D RESAMPLING (INTENDED VOXEL 4MM ISOTROPIC)
# https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dresample.html
# http://imaging.mrc-cbu.cam.ac.uk/imaging/Introduction_to_fMRI_2010?action=AttachFile&do=get&target=Intro_fMRI_2010_01_preprocessing.pdf
# https://www.mathworks.com/help/images/ref/imresize3.html
# https://stackoverflow.com/questions/30459950/downsampling-of-fmri-image-with-fsl
# flirt -in PEER1.nii.gz -ref PEER1.nii.gz -out PEER1_resampled -applyisoxfm 4 COMMAND ON NED TERMINAL FOR FSL
# Participant A has low intensity values, Participant C has high intensity values

train_vectors = np.asarray(listed)
test_vectors = np.asarray(listed_testing)

averaged_train = []
averaged_test = []
for num in [i*5 for i in range(27)]:

    averaged_train.append(np.average(train_vectors[num:num+5], axis=0))
    averaged_test.append(np.average(test_vectors[num:num+5], axis=0))

train_vectors = np.asarray(averaged_train)
test_vectors = np.asarray(averaged_test)

# Create np array that contains all fixation locations, separated by x and y coordinates

fixations = pd.read_csv('stim_vals.csv')
x_targets = np.repeat(np.array(fixations['pos_x']), 1)*monitor_width/2
y_targets = np.repeat(np.array(fixations['pos_y']), 1)*monitor_height/2

# # For visualization
# for num in [0, 50]:
#     visual = image.index_img(testing, num)
#     slice_data = visual.get_data()[:, :, 15:30]
#     plotting.plot_stat_map(visual)
#     plotting.show()

# Build classifiers with variable hyperparameters

classifier = SVR(kernel='linear', C=100, epsilon=.001)
# Validationg
validation_x = classifier.fit(train_vectors, x_targets).predict(train_vectors)
validation_y = classifier.fit(train_vectors, y_targets).predict(train_vectors)
# Testing
predicted_x = classifier.fit(train_vectors, x_targets).predict(test_vectors)
predicted_y = classifier.fit(train_vectors, y_targets).predict(test_vectors)

GS_model = SVR()
parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [100, 200, 300, 400, 500, 1000, 2500, 5000, 10000],
              'epsilon': [.01, .001]}
clfx = GridSearchCV(GS_model, parameters)
clfx.fit(train_vectors, x_targets)
clfy = GridSearchCV(GS_model, parameters)
clfy.fit(train_vectors, y_targets)

predicted_x = clfx.predict(test_vectors)
predicted_y = clfy.predict(test_vectors)

# Plot SVR predictions against targets

plt.figure()

for num in [i*5 for i in range(27)]:
    plt.scatter(predicted_x[num:num+5], predicted_y[num:num+5], alpha=.5)

plt.scatter(x_targets, y_targets, color='k', marker='x', alpha=1)
plt.xlabel('x-position')
plt.ylabel('y-position')
plt.title('Support Vector Regression - PEER')
plt.show()
