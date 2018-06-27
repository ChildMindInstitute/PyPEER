import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVR

data_dir = '/path/to/data/folder'
eye_mask_path = '/eye/mask/path'

monitor_width = 1680
monitor_height = 1050

fixations = pd.read_csv('stim_vals.csv')
x_targets = np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2
y_targets = np.repeat(np.array(fixations['pos_y']), 1) * monitor_height / 2


def mean_center_var_norm(data):

    volumes = data.shape[3]

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                vmean = np.mean(np.array(data[x, y, z, :]))
                vstdev = np.std(np.array(data[x, y, z, :]))

                for time in range(volumes):
                    if vstdev != 0:
                        data[x, y, z, time] = (float(data[x, y, z, time]) - float(vmean))/vstdev
                    else:
                        data[x, y, z, time] = float(data[x, y, z, time]) - float(vmean)

    return data


def gs_regress(data):

    global_mask = np.array(eye_mask_path, dtype=bool)

    regressor_map = {'constant': np.ones((data.shape[3], 1))}
    regressor_map['global'] = data[global_mask].mean(0)

    X = np.zeros((data.shape[3], 1))
    csv_filename = ''

    for rname, rval in regressor_map.items():
        X = np.hstack((X, rval.reshape(rval.shape[0], -1)))
        csv_filename += '_' + rname

    X = X[:, 1:]

    Y = data[global_mask].T
    B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    Y_res = Y - X.dot(B)

    data[global_mask] = Y_res.T

    return data


def process_data(filename, gsr_status=False, TR_per_target=5):

    train_data_time_point = []
    train_data_averaged = []

    file_dir = os.path.join(data_dir, filename)

    train_data = nib.load(file_dir).get_data()

    train_data = mean_center_var_norm(train_data)

    if gsr_status:
        train_data = gs_regress(train_data)

    for time_point in range(train_data.shape[3]):
        train_data = train_data[:, :, :, time_point]
        train_data = np.array(train_data.ravel())
        train_data_time_point.append(train_data)

    for calibration_target in range(train_data.shape[3] / TR_per_target):
        train_data_averaged.append(np.average(train_data_time_point[calibration_target * 5:5*(1+calibration_target)], axis=0))

    return train_data_averaged


def train_svr_model(fmri_files = ['train_file1', 'train_file2', 'train_file3']):

    processed_files = []

    for filename in fmri_files:

        processed_data = process_data(filename, gsr_status=False, TR_per_target=5)
        processed_files.append(processed_data)

    training_set = [[x for x in file] for file in processed_files]

    x_model = SVR(kernel='linear', C=100, epsilon=.01, verbose=2)
    y_model = SVR(kernel='linear', C=100, epsilon=.01, verbose=2)
    x_model.fit(training_set, x_targets)
    y_model.fit(training_set, y_targets)

    # TODO Save the model





