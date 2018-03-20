import os
import os.path
import csv
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVR
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


def gs_regress(data, xb, xe, yb, ye, zb, ze):

    global_mask = np.zeros([data.shape[0], data.shape[1], data.shape[2]])

    for x in range(int(global_mask.shape[0])):
        for y in range(int(global_mask.shape[1])):
            for z in range(int(global_mask.shape[2])):
                if x in range(xb, xe) and y in range(yb, ye) and z in range(zb, ze):

                    global_mask[x, y, z] = 1

    global_mask = np.array(global_mask, dtype=bool)

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


def remove_out(group):

    temp = params[group][:22].tolist()
    temp = [float(x) for x in temp]
    temp_mean = np.array(temp).mean()
    temp_stdv = np.array(temp).std()
    cap = temp_mean + 3*temp_stdv

    modified_group = [x for x in temp if 0 < x < cap]

    return modified_group


def create_model(train_vectors, x_targets, y_targets):

    startTime = datetime.now()

    print('Making model')

    x_model = SVR(kernel='linear', C=100, epsilon=.01, verbose=2)
    y_model = SVR(kernel='linear', C=100, epsilon=.01, verbose=2)
    x_model.fit(train_vectors, x_targets)
    y_model.fit(train_vectors, y_targets)

    print('\nRuntime: ' + str(datetime.now() - startTime))

    return x_model, y_model


def predict_fixations(x_model, y_model, test_vectors):

    predicted_x = x_model.predict(test_vectors)
    print('\nCompleted x model predictions')
    predicted_y = y_model.predict(test_vectors)
    print('\nCompleted y model predictions')

    return predicted_x, predicted_y


def scatter_plot(name, x_targets, y_targets, predicted_x, predicted_y, plot=False, save=False):

    if plot:

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
                plt.scatter(predicted_x[nums:nums + 5], predicted_y[nums:nums + 5], s=5)

            elif num == 18 or num == 25:

                plt.subplot(5, 5, 1)
                axes = plt.gca()
                axes.set_xlim([-1500, 1500])
                axes.set_ylim([-1200, 1200])
                plt.scatter(x_targets[num], y_targets[num], color='k', marker='x')
                plt.scatter(predicted_x[nums:nums + 5], predicted_y[nums:nums + 5], s=5)

            else:
                continue

        if save:

            plt.savefig(os.path.join(output_path, name + '.png'), bbox_inches='tight', dpi=600)
            plt.show()


def data_processing(scan_count, train_vectors1, train_vectors2):

    averaged_train1 = []
    averaged_train2 = []

    for num in [i*5 for i in range(int(len(train_vectors1)/5))]:

        averaged_train1.append(np.average(train_vectors1[num:num+5], axis=0))

        if scan_count == 3:
            averaged_train2.append(np.average(train_vectors2[num:num+5], axis=0))

    if scan_count == 2:
        train_vectors = np.asarray(averaged_train1)
        print('Has ' + str(scan_count) + ' valid PEER scans')
    elif scan_count == 3:
        train_vectors = np.asarray(averaged_train1 + averaged_train2)
        print('Has ' + str(scan_count) + ' valid PEER scans')

    return train_vectors


def rmse_hist(save=False):

    params = pd.read_csv('subj_params.csv', index_col='subject', dtype=object)
    params = params[params['scan_count'] == str(3)]

    x_1 = remove_out('x_error_reg')
    y_1 = remove_out('y_error_reg')
    x_b = remove_out('x_error_within')
    y_b = remove_out('y_error_within')

    xbins = np.histogram(np.hstack((x_1, x_b)), bins=15)[1]
    ybins = np.histogram(np.hstack((y_1, y_b)), bins=15)[1]

    plt.figure()
    plt.hist(x_1, xbins, color='r', alpha=.5, label='Subject')
    plt.hist(x_b, xbins, color='g', alpha=.5, label='General')
    plt.title('RMSE distribution in the x-direction')
    plt.ylabel('Number of participants')
    plt.xlabel('RMSE (x-direction)')
    plt.legend()

    if save:

        plt.savefig(os.path.join(output_path, 'x_dir_histogram'), bbox_inches='tight', dpi=600)

    plt.figure()
    plt.hist(y_1, ybins, color='r', alpha=.5, label='Subject')
    plt.hist(y_b, ybins, color='g', alpha=.5, label='General')
    plt.legend()
    plt.title('RMSE distribution in the y-direction')
    plt.ylabel('Number of participants')
    plt.xlabel('RMSE (y-direction)')

    if save:

        plt.savefig(os.path.join(output_path, 'y_dir_histogram'), bbox_inches='tight', dpi=600)

    plt.show()


def axis_plot(fixations, predicted_x, predicted_y, subj, train_sets=1):

    x_targets = np.repeat(np.array(fixations['pos_x']), 5*train_sets) * monitor_width / 2
    y_targets = np.repeat(np.array(fixations['pos_y']), 5*train_sets) * monitor_height / 2

    time_series = range(0, len(predicted_x))

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.ylabel('Horizontal position')
    # plt.plot(time_series, x_targets, '.-', color='k')
    # plt.plot(time_series, predicted_x, '.-', color='b')
    # plt.subplot(2, 1, 2)
    # plt.ylabel('Vertical position')
    # plt.xlabel('TR')
    # # plt.title('Participant ' + str(subj))
    # plt.plot(time_series, y_targets, '.-', color='k')
    # plt.plot(time_series, predicted_y, '.-', color='b')
    # # plt.savefig(os.path.join(output_path, subj + 'peer.png'), bbox_inches='tight', dpi=600)
    # plt.show()

    return x_targets, y_targets


def update_subjects(site='RU'):

    # Include new participants from os.listdir()

    resample_path = '/data2/Projects/Jake/Human_Brain_Mapping/'

    params = pd.read_csv('peer_didactics.csv', index_col='subject', dtype=object)
    sub_ref = params.index.values.tolist()

    with open('peer_didactics.csv', 'a') as updated_params:
        writer = csv.writer(updated_params)

        for subject in os.listdir(resample_path):
            if (any(subject in x for x in sub_ref)) and ('txt' not in subject):
                print(subject + ' is already in subj_params.csv')
            elif 'txt' not in subject:
                writer.writerow([subject])
                print('New participant ' + subject + ' was added')

    # Include site, scan_count

    if site == 'RU':
        qap_path = '/data2/HBNcore/CMI_HBN_Data/MRI/RU/QAP/qap_functional_temporal.csv'
    elif site == 'CBIC':
        qap_path = '/data2/HBNcore/CMI_HBN_Data/MRI/CBIC/QAP/qap_functional_temporal.csv'

    qap = pd.read_csv(qap_path, dtype=object)
    qap['Participant'] = qap['Participant'].str.replace('_', '-')

    params = pd.read_csv('peer_didactics.csv', index_col='subject', dtype=object)
    sub_list = params.index.values.tolist()

    for sub in sub_list:

        print('Obtaining site and number of complete calibration scans for subject ' + str(sub))

        params = pd.read_csv('peer_didactics.csv', index_col='subject', dtype=object)

        scan_count = int(os.path.isfile(resample_path + sub + '/peer1_eyes_sub.nii.gz')) + \
                     int(os.path.isfile(resample_path + sub + '/peer2_eyes_sub.nii.gz')) + \
                     int(os.path.isfile(resample_path + sub + '/peer3_eyes_sub.nii.gz'))

        params.loc[sub, 'scan_count'] = scan_count
        params.loc[sub, 'site'] = site
        params.to_csv('peer_didactics.csv')

    # Include motion measures

    for sub in sub_list:

        print('Obtaining motion measures for subject ' + str(sub))

        params = pd.read_csv('peer_didactics.csv', dtype=object)

        scan_count = int(params[params.subject == sub].scan_count)

        if scan_count == 3:

            fd1 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_1')]['RMSD (Mean)'])
            fd2 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_2')]['RMSD (Mean)'])
            fd3 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_3')]['RMSD (Mean)'])
            dv1 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_1')]['Std. DVARS (Mean)'])
            dv2 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_1')]['Std. DVARS (Mean)'])
            dv3 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_1')]['Std. DVARS (Mean)'])

            fdm = np.average([fd1, fd2, fd3])
            dvm = np.average([dv1, dv2, dv3])

        elif scan_count == 2:

            fd1 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_1')]['RMSD (Mean)'])
            fd2 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_2')]['RMSD (Mean)'])
            dv1 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_1')]['Std. DVARS (Mean)'])
            dv2 = float(qap[(qap.Participant == sub) & (qap.Series == 'func_peer_run_1')]['Std. DVARS (Mean)'])

            fdm = np.average([fd1, fd2])
            dvm = np.average([dv1, dv2])


        else:
            print('Not enough scans to update motion measures')

        params = params.set_index('subject')
        params.loc[sub, 'mean_fd'] = fdm
        params.loc[sub, 'dvars'] = dvm
        params.to_csv('peer_didactics.csv')


