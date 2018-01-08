# #############################################################################
# PEER testing for fMRI data

# Screen size: 1680 x 1050 - from follow_the_dot_lastrun.py in ~/Desktop/HBN_Peer

# Participant A: 10:18
# Participant B: 3:8
# Participant C: 2:8
# Participant D: 5:11

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

img = nib.load('aPEER1_resampled.nii.gz')
data = img.get_data()
testing = nib.load('aPEER3_resampled.nii.gz')
testing_data = testing.get_data()

# Vectorize data into single np array

listed = []
listed_testing = []

x_begin_slice = 15
x_end_slice = 40
y_begin_slice = 2
y_end_slice = 12
z_begin_slice = 10
z_end_slice = 18

for tr in range(int(data.shape[3])): # training with 5TRs from each fixation (averaged or individual)
# for tr in [i*5 for i in range(27)]: # training with one TR from each fixation

    tr_data = data[x_begin_slice:x_end_slice, y_begin_slice:y_end_slice, z_begin_slice:z_end_slice, tr]
    te_data = testing_data[x_begin_slice:x_end_slice, y_begin_slice:y_end_slice, z_begin_slice:z_end_slice, tr]

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
# test_vectors = np.asarray(averaged_test)

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

# classifier = SVR(kernel='linear', C=100, epsilon=.001)
# # Validationg
# validation_x = classifier.fit(train_vectors, x_targets).predict(train_vectors)
# validation_y = classifier.fit(train_vectors, y_targets).predict(train_vectors)
# # Testing
# predicted_x = classifier.fit(train_vectors, x_targets).predict(test_vectors)
# predicted_y = classifier.fit(train_vectors, y_targets).predict(test_vectors)

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

RMS_valuesx = []
RMS_valuesy = []

# for num in range(0, 27):
for num in range(0, 5):

    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-1200, 1200])
    axes.set_ylim([-600, 600])

    nums = num * 5

    plt.scatter(x_targets[num], y_targets[num], color='k')
    RMS_tempx = []
    RMS_tempy = []
    for number in [nums, nums+1, nums+2, nums+3, nums+4]:
        plt.scatter(predicted_x[number], predicted_y[number])
        print(number)
        RMS_tempx.append((predicted_x[number]-x_targets[num])**2)
        RMS_tempy.append((predicted_y[number]-y_targets[num])**2)

    RMS_valuesx.append((np.average(np.array(RMS_tempx))/5)**.5)
    RMS_valuesy.append((np.average(np.array(RMS_tempy))/5)**.5)
    print((RMS_valuesx, RMS_valuesy))

    plt.show()

# Used when each TR is a training sample
# for num in [i*5 for i in range(27)]:
#     plt.scatter(predicted_x[num:num+5], predicted_y[num:num+5], alpha=.5)

# Used when 5TR averaging
# for num in range(27):
#     plt.scatter(predicted_x[num], predicted_y[num], alpha=.5)

# Plot all targets
# plt.scatter(x_targets, y_targets, color='k', marker='x', alpha=1)
# plt.scatter(predicted_x, predicted_y)

# Plot individual targets
# for num in range(len(x_targets)):
#     plt.figure()
#     axes = plt.gca()
#     axes.set_xlim([-1200, 1200])
#     axes.set_ylim([-600, 600])
#     plt.scatter(x_targets[num], y_targets[num])
#     plt.scatter(predicted_x[num], predicted_y[num])
#     print((x_targets[num] - predicted_x[num]) / x_targets[num])
#     print((y_targets[num] - predicted_y[num]) / y_targets[num])
#
# plt.xlabel('x-position')
# plt.ylabel('y-position')
# plt.title('Support Vector Regression - PEER')
# plt.show()
