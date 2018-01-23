import os
import csv
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

data_path = '/home/json/Desktop/PEER_bash/'
output_path = '/home/json/Desktop/peer/Figures'

monitor_width = 1680
monitor_height = 1050

# #############################################################################
# Update subject parameters sheet with new participants

params = pd.read_csv('subj_params.csv', index_col='subject', dtype=object)
sub_ref = params.index.values.tolist()

x_b = 12
x_e = 40
y_b = 35
y_e = 50
z_b = 2
z_e = 13

subj_list = []

with open('subj_params.csv', 'a') as updated_params:
    writer = csv.writer(updated_params)

    for subject in os.listdir(data_path):
        if any(subject in x for x in sub_ref) and 'txt' not in subject:
            print(subject + ' is already in subj_params.csv')
        elif 'txt' not in subject:
            writer.writerow([subject, x_b, x_e, y_b, y_e, z_b, z_e])
            print('New participant ' + subject + ' was added')
            subj_list.append(subject)

# #############################################################################
# Import data

params = pd.read_csv('subj_params.csv', index_col='subject', dtype=object)

for set in subj_list:

    x_begin_slice = int(params.loc[set, 'x_start'])
    x_end_slice = int(params.loc[set, 'x_end'])
    y_begin_slice = int(params.loc[set, 'y_start'])
    y_end_slice = int(params.loc[set, 'y_end'])
    z_begin_slice = int(params.loc[set, 'z_start'])
    z_end_slice = int(params.loc[set, 'z_end'])

    print('Beginning analysis on participant ' + set)

    training1 = nib.load(data_path + set + '/PEER1_resampled.nii.gz')
    training1_data = training1.get_data()
    testing = nib.load(data_path + set + '/PEER2_resampled.nii.gz')
    testing_data = testing.get_data()

    try:

        training2 = nib.load(data_path + set + '/PEER3_resampled.nii.gz')
        training2_data = training2.get_da
        scan_count = 3

    except:

        scan_count = 2

    # #############################################################################
    # Vectorize data into single np array

    listed1 = []
    listed2 = []
    listed_testing = []

    for tr in range(int(training1_data.shape[3])):

        tr_data1 = training1_data[x_begin_slice:x_end_slice, y_begin_slice:y_end_slice, z_begin_slice:z_end_slice, tr]
        vectorized1 = np.array(tr_data1.ravel())
        listed1.append(vectorized1)

        if scan_count == 3:

            tr_data2 = training2_data[x_begin_slice:x_end_slice, y_begin_slice:y_end_slice, z_begin_slice:z_end_slice, tr]
            vectorized2 = np.array(tr_data2.ravel())
            listed2.append(vectorized2)

        te_data = testing_data[x_begin_slice:x_end_slice, y_begin_slice:y_end_slice, z_begin_slice:z_end_slice, tr]
        vectorized_testing = np.array(te_data.ravel())
        listed_testing.append(vectorized_testing)

    train_vectors1 = np.asarray(listed1)
    test_vectors = np.asarray(listed_testing)

    if scan_count == 3:
        train_vectors2 = np.asarray(listed2)

    # #############################################################################
    # Averaging training signal

    averaged_train1 = []
    averaged_train2 = []
    averaged_test = []

    for num in [i*5 for i in range(27)]:

        averaged_train1.append(np.average(train_vectors1[num:num+5], axis=0))

        if scan_count == 3:
            averaged_train2.append(np.average(train_vectors2[num:num+5], axis=0))

    if scan_count == 2:
        train_vectors = np.asarray(averaged_train1)
        print('Has ' + str(scan_count) + ' valid PEER scans')
    elif scan_count == 3:
        train_vectors = np.asarray(averaged_train1 + averaged_train2)
        print('Has ' + str(scan_count) + ' valid PEER scans')

    # #############################################################################
    # Import coordinates for fixations

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1)*monitor_width/2, scan_count-1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1)*monitor_height/2, scan_count-1)

    # #############################################################################
    # Create SVR Model

    # GridSearch Model
    GS_model = SVR(kernel='linear')
    parameters = {'kernel': ('linear', 'poly'), 'C': [100, 1000, 2500, 5000, 10000],
                  'epsilon': [.01, .005, .001]}
    clfx = GridSearchCV(GS_model, parameters)
    clfx.fit(train_vectors, x_targets)
    clfy = GridSearchCV(GS_model, parameters)
    clfy.fit(train_vectors, y_targets)
    predicted_x = clfx.predict(test_vectors)
    predicted_y = clfy.predict(test_vectors)

    # #############################################################################
    # Plot SVR predictions against targets

    plt.figure(figsize=(10, 10))
    subplot_i = 0

    for num in range(27):

        nums = num * 5

        if num not in [18, 25]:

            subplot_i += 1
            plt.subplot(5, 5, subplot_i)
            axes = plt.gca()
            axes.set_xlim([-1500, 1500])
            axes.set_ylim([-1200, 1200])
            plt.scatter(x_targets[num], y_targets[num], color='k', marker='x')
            plt.scatter(predicted_x[nums:nums+5], predicted_y[nums:nums+5], s=5)

        elif num == 18 or num == 25:

            plt.subplot(5, 5, 1)
            axes = plt.gca()
            axes.set_xlim([-1500, 1500])
            axes.set_ylim([-1200, 1200])
            plt.scatter(x_targets[num], y_targets[num], color='k', marker='x')
            plt.scatter(predicted_x[nums:nums+5], predicted_y[nums:nums+5], s=5)

        else:
            continue

    plt.savefig(os.path.join(output_path, set + '.png'), bbox_inches='tight', dpi=600)
    # plt.show()

    print('Completed participant ' + set)

    # ###############################################################################
    # Get error meaasurements

    x_res = []
    y_res = []

    for num in range(27):

        nums = num * 5

        for values in range(5):

            error_x = (abs(x_targets[num] - predicted_x[nums + values]))**2
            error_y = (abs(y_targets[num] - predicted_y[nums + values]))**2
            x_res.append(error_x)
            y_res.append(error_y)

    x_error = np.sqrt(np.sum(np.array(x_res))/135)
    y_error = np.sqrt(np.sum(np.array(y_res))/135)

    params.loc[set, 'x_error'] = x_error
    params.loc[set, 'y_error'] = y_error
    params.to_csv('subj_params.csv')

# #############################################################################
# Visualize error vs motion

params = pd.read_csv('subj_params.csv', index_col='subject', dtype=object)

num_part = 15

x_error_list = params.loc[:, 'x_error'][:num_part].tolist()
y_error_list = params.loc[:, 'y_error'][:num_part].tolist()
mean_fd_list = params.loc[:, 'mean_fd'][:num_part].tolist()
dvars_list = params.loc[:, 'dvars'][:num_part].tolist()

x_error_list = np.array([float(x) for x in x_error_list])
y_error_list = np.array([float(x) for x in y_error_list])
mean_fd_list = np.array([float(x) for x in mean_fd_list])
dvars_list = np.array([float(x) for x in dvars_list])

m1, b1 = np.polyfit(mean_fd_list, x_error_list, 1)
m2, b2 = np.polyfit(mean_fd_list, y_error_list, 1)
m3, b3 = np.polyfit(dvars_list, x_error_list, 1)
m4, b4 = np.polyfit(dvars_list, y_error_list, 1)

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.title('mean_fd vs. x_RMS')
plt.scatter(mean_fd_list, x_error_list, s=5)
plt.plot(mean_fd_list, m1*mean_fd_list + b1, '-', color='r')
plt.subplot(2, 2, 2)
plt.title('mean_fd vs. y_RMS')
plt.scatter(mean_fd_list, y_error_list, s=5)
plt.plot(mean_fd_list, m2*mean_fd_list + b2, '-', color='r')
plt.subplot(2, 2, 3)
plt.title('dvars vs. x_RMS')
plt.scatter(dvars_list, x_error_list, s=5)
plt.plot(dvars_list, m3*dvars_list + b3, '-', color='r')
plt.subplot(2, 2, 4)
plt.title('dvars vs. y_RMS')
plt.scatter(dvars_list, y_error_list, s=5)
plt.plot(dvars_list, m4*dvars_list + b4, '-', color='r')
plt.show()