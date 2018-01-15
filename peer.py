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

set = 'b'

training1 = nib.load(set + 'PEER1_resampled.nii.gz')
training2 = nib.load(set + 'PEER3_resampled.nii.gz')
testing = nib.load(set + 'PEER2_resampled.nii.gz')
training1_data = training1.get_data()
training2_data = training2.get_data()
testing_data = testing.get_data()

if set == 'a':
    x_begin_slice = 15
    x_end_slice = 40
    y_begin_slice = 2
    y_end_slice = 12
    z_begin_slice = 10
    z_end_slice = 18
elif set == 'b':
    x_begin_slice = 15
    x_end_slice = 39
    y_begin_slice = 6
    y_end_slice = 13
    z_begin_slice = 3
    z_end_slice = 8
elif set == 'c':
    x_begin_slice = 10
    x_end_slice = 38
    y_begin_slice = 6
    y_end_slice = 13
    z_begin_slice = 2
    z_end_slice = 8

# Vectorize data into single np array

listed1 = []
listed2 = []
listed_testing = []

for tr in range(int(training1_data.shape[3])): # training with 5TRs from each fixation (averaged or individual)
# for tr in [i*5 for i in range(27)]: # training with one TR from each fixation

    tr_data1 = training1_data[x_begin_slice:x_end_slice, y_begin_slice:y_end_slice, z_begin_slice:z_end_slice, tr]
    tr_data2 = training2_data[x_begin_slice:x_end_slice, y_begin_slice:y_end_slice, z_begin_slice:z_end_slice, tr]
    te_data = testing_data[x_begin_slice:x_end_slice, y_begin_slice:y_end_slice, z_begin_slice:z_end_slice, tr]
    vectorized1 = np.array(tr_data1.ravel())
    vectorized2 = np.array(tr_data2.ravel())
    vectorized_testing = np.array(te_data.ravel())
    listed1.append(vectorized1)
    listed2.append(vectorized2)
    listed_testing.append(vectorized_testing)

# CHECK AFNI, FSL, NILEARN FOR 3D RESAMPLING (INTENDED VOXEL 4MM ISOTROPIC)
# https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dresample.html
# http://imaging.mrc-cbu.cam.ac.uk/imaging/Introduction_to_fMRI_2010?action=AttachFile&do=get&target=Intro_fMRI_2010_01_preprocessing.pdf
# https://www.mathworks.com/help/images/ref/imresize3.html
# https://stackoverflow.com/questions/30459950/downsampling-of-fmri-image-with-fsl
# flirt -in PEER1.nii.gz -ref PEER1.nii.gz -out PEER1_resampled -applyisoxfm 4 COMMAND ON NED TERMINAL FOR FSL

train_vectors1 = np.asarray(listed1)
train_vectors2 = np.asarray(listed2)
test_vectors = np.asarray(listed_testing)

# Averaging training signal

averaged_train1 = []
averaged_train2 = []
averaged_test = []
for num in [i*5 for i in range(27)]:

    averaged_train1.append(np.average(train_vectors1[num:num+5], axis=0))
    averaged_train2.append(np.average(train_vectors2[num:num+5], axis=0))
    # averaged_test.append(np.average(test_vectors[num:num+5], axis=0))

if averaged_train2 == []:
    train_vectors = np.asarray(averaged_train1)
    print('just one')
else:
    train_vectors = np.asarray(averaged_train1 + averaged_train2)
    print('concatenated')

# test_vectors = np.asarray(averaged_test)

# Create np array that contains all fixation locations, separated by x and y coordinates

fixations = pd.read_csv('stim_vals.csv')
# x_targets = np.repeat(np.array(fixations['pos_x']), 1)*monitor_width/2
# y_targets = np.repeat(np.array(fixations['pos_y']), 1)*monitor_height/2
x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1)*monitor_width/2, 2)
y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1)*monitor_height/2, 2)

# # For visualization
# for num in [0, 50]:
#     visual = image.index_img(testing, num)
#     slice_data = visual.get_data()[:, :, 15:30]
#     plotting.plot_stat_map(visual)
#     plotting.show()

# # Manual Model
# clfx = SVR(kernel='linear', C=10000, epsilon=.0001)
# clfx.fit(train_vectors, x_targets)
# clfy = SVR(kernel='linear', C=10000, epsilon=.0001)
# clfy.fit(train_vectors, y_targets)
# predicted_x = clfx.predict(test_vectors)
# predicted_y = clfy.predict(test_vectors)

# GridSearch Model
GS_model = SVR(kernel='linear')
parameters = {'kernel': ('linear', 'sigmoid'), 'C': [10, 50, 100, 200, 300,
                                                                    400, 500, 1000, 2500, 5000, 10000],
              'epsilon': [.01, .005, .001]}
clfx = GridSearchCV(GS_model, parameters)
clfx.fit(train_vectors, x_targets)
clfy = GridSearchCV(GS_model, parameters)
clfy.fit(train_vectors, y_targets)

predicted_x = clfx.predict(test_vectors)
predicted_y = clfy.predict(test_vectors)

# Plot SVR predictions against targets

plt.figure(figsize=(10, 10))
subplot_i = 0

for num in range(27):

    nums = num * 5

    if num not in [18, 25]:

        subplot_i += 1
        plt.subplot(5, 5, subplot_i)
        axes = plt.gca()
        axes.set_xlim([-1200, 1200])
        axes.set_ylim([-900, 900])
        plt.scatter(x_targets[num], y_targets[num], color='k', marker='x')
        plt.scatter(predicted_x[nums:nums+5], predicted_y[nums:nums+5], s=5)

    elif num == 18 or num == 25:

        plt.subplot(5, 5, 1)
        axes = plt.gca()
        axes.set_xlim([-1200, 1200])
        axes.set_ylim([-900, 900])
        plt.scatter(x_targets[num], y_targets[num], color='k', marker='x')
        plt.scatter(predicted_x[nums:nums+5], predicted_y[nums:nums+5], s=5)

    else:
        continue

plt.show()
