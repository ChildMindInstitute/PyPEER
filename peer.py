import os
import csv
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVR
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from datetime import datetime


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

    # x_save = pickle.dumps(x_model)
    # joblib.dump(x_save, 'x_model_1_resample50.pkl')
    # y_save = pickle.dumps(y_model)
    # joblib.dump(y_save, 'y_model_1_resample50.pkl')
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

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel('Horizontal position')
    plt.plot(time_series, x_targets, '.-', color='k')
    plt.plot(time_series, predicted_x, '.-', color='b')
    plt.subplot(2, 1, 2)
    plt.ylabel('Vertical position')
    plt.xlabel('TR')
    plt.title('Participant ' + str(subj))
    plt.plot(time_series, y_targets, '.-', color='k')
    plt.plot(time_series, predicted_y, '.-', color='b')
    # plt.savefig(os.path.join(output_path, 'peer_3.png'), bbox_inches='tight', dpi=600)
    plt.show()

# data_path = '/data2/Projects/Jake/Registration_Test/'
data_path = '/data2/Projects/Jake/CBIC/'
qap_path_RU = '/data2/HBNcore/CMI_HBN_Data/MRI/RU/QAP/qap_functional_temporal.csv'
qap_path_CBIC = '/data2/HBNcore/CMI_HBN_Data/MRI/CBIC/QAP/qap_functional_temporal.csv'
output_path = '/home/json/Desktop/peer/Figures'

monitor_width = 1680
monitor_height = 1050

# #############################################################################
# Update subject parameters sheet with new participants

params = pd.read_csv('subj_params.csv', index_col='subject', dtype=object)
sub_ref = params.index.values.tolist()

qap = pd.read_csv(qap_path_RU, dtype=object)
qap['Participant'] = qap['Participant'].str.replace('_', '-')

x_b = 12; x_e = 40; y_b = 35; y_e = 50; z_b = 2; z_e = 13

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

full_set = params.index.values.tolist()


def standard_peer(subject_list, gsr=True, update=False):

    for name in subject_list:

        xb = int(params.loc[name, 'x_start']); xe = int(params.loc[name, 'x_end'])
        yb = int(params.loc[name, 'y_start']); ye = int(params.loc[name, 'y_end'])
        zb = int(params.loc[name, 'z_start']); ze = int(params.loc[name, 'z_end'])

        try:

            print('Beginning analysis on participant ' + name)

            training1 = nib.load(data_path + name + '/PEER1_resampled.nii.gz')
            training1_data = training1.get_data()
            testing = nib.load(data_path + name + '/PEER2_resampled.nii.gz')
            testing_data = testing.get_data()

            try:

                training2 = nib.load(data_path + name + '/PEER3_resampled.nii.gz')
                training2_data = training2.get_data()
                scan_count = 3

            except:

                scan_count = 2

            # #############################################################################
            # Global Signal Regression

            print('starting gsr')

            if gsr:

                print('entered loop')

                if scan_count == 2:

                    print('count = 2')

                    training1_data = gs_regress(training1_data, xb, xe, yb, ye, zb, ze)
                    testing_data = gs_regress(testing_data, xb, xe, yb, ye, zb, ze)

                elif scan_count == 3:

                    print('count = 3')

                    training1_data = gs_regress(training1_data, xb, xe, yb, ye, zb, ze)
                    training2_data = gs_regress(training2_data, xb, xe, yb, ye, zb, ze)
                    testing_data = gs_regress(testing_data, xb, xe, yb, ye, zb, ze)

            # #############################################################################
            # Vectorize data into single np array

            listed1 = []
            listed2 = []
            listed_testing = []

            print('beginning vectors')

            for tr in range(int(training1_data.shape[3])):

                tr_data1 = training1_data[xb:xe, yb:ye, zb:ze, tr]
                vectorized1 = np.array(tr_data1.ravel())
                listed1.append(vectorized1)

                if scan_count == 3:

                    tr_data2 = training2_data[xb:xe, yb:ye, zb:ze, tr]
                    vectorized2 = np.array(tr_data2.ravel())
                    listed2.append(vectorized2)

                te_data = testing_data[xb:xe, yb:ye, zb:ze, tr]
                vectorized_testing = np.array(te_data.ravel())
                listed_testing.append(vectorized_testing)

            train_vectors1 = np.asarray(listed1)
            test_vectors = np.asarray(listed_testing)

            if scan_count == 3:
                train_vectors2 = np.asarray(listed2)

            elif scan_count == 2:
                train_vectors2 = []

            # #############################################################################
            # Averaging training signal

            print('average vectors')

            train_vectors = data_processing(scan_count, train_vectors1, train_vectors2)

            # #############################################################################
            # Import coordinates for fixations

            print('importing fixations')

            fixations = pd.read_csv('stim_vals.csv')
            x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1)*monitor_width/2, scan_count-1)
            y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1)*monitor_height/2, scan_count-1)

            # #############################################################################
            # Create SVR Model

            predicted_x, predicted_y = create_model(train_vectors, test_vectors, x_targets, y_targets)

            # #############################################################################
            # Plot SVR predictions against targets

            # scatter_plot(name, x_targets, y_targets, predicted_x, predicted_y, plot=True, save=False)
            axis_plot(fixations, predicted_x, predicted_y, subj)

            print('Completed participant ' + name)

            # ###############################################################################
            # Get error measurements

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

            params.loc[name, 'x_error_gsr'] = x_error
            params.loc[name, 'y_error_gsr'] = y_error
            params.loc[name, 'scan_count'] = scan_count

            if update:
                params.to_csv('subj_params.csv')

            if scan_count == 3:

                try:

                    fd1 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_1']['RMSD (Mean)'])
                    fd2 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_2']['RMSD (Mean)'])
                    fd3 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_3']['RMSD (Mean)'])
                    dv1 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_1']['Std. DVARS (Mean)'])
                    dv2 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_2']['Std. DVARS (Mean)'])
                    dv3 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_3']['Std. DVARS (Mean)'])

                    fd_score = np.average([fd1, fd2, fd3])
                    dvars_score = np.average([dv1, dv2, dv3])
                    params.loc[name, 'mean_fd'] = fd_score
                    params.loc[name, 'dvars'] = dvars_score

                except:

                    print('Participant not found in QAP')

            elif scan_count == 2:

                try:

                    fd1 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_1']['RMSD (Mean)'])
                    fd2 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_2']['RMSD (Mean)'])
                    dv1 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_1']['Std. DVARS (Mean)'])
                    dv2 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_2']['Std. DVARS (Mean)'])

                    fd_score = np.average([[fd1, fd2]])
                    dvars_score = np.average([dv1, dv2])
                    params.loc[name, 'mean_fd'] = fd_score
                    params.loc[name, 'dvars'] = dvars_score

                except:

                    print('Participant not found in QAP')

            if update:

                params.to_csv('subj_params.csv')

        except:

            print('Error processing participant')

standard_peer(['sub-5437909'], gsr=True, update=False)


def icc_peer(subject_list, gsr=False, update=False, scan=1):

    for name in subject_list:

        xb = int(params.loc[name, 'x_start']); xe = int(params.loc[name, 'x_end'])
        yb = int(params.loc[name, 'y_start']); ye = int(params.loc[name, 'y_end'])
        zb = int(params.loc[name, 'z_start']); ze = int(params.loc[name, 'z_end'])

        try:

            print('Beginning analysis on participant ' + name)

            training1 = nib.load(data_path + name + '/PEER' + str(scan) + '_resampled.nii.gz')
            training1_data = training1.get_data()
            testing = nib.load(data_path + name + '/PEER2_resampled.nii.gz')
            testing_data = testing.get_data()
            scan_count = 2

            print('starting gsr')

            if gsr:

                training1_data = gs_regress(training1_data, xb, xe, yb, ye, zb, ze)
                testing_data = gs_regress(testing_data, xb, xe, yb, ye, zb, ze)

            listed1 = []
            listed_testing = []

            print('beginning vectors')

            for tr in range(int(training1_data.shape[3])):

                tr_data1 = training1_data[xb:xe, yb:ye, zb:ze, tr]
                vectorized1 = np.array(tr_data1.ravel())
                listed1.append(vectorized1)
                te_data = testing_data[xb:xe, yb:ye, zb:ze, tr]
                vectorized_testing = np.array(te_data.ravel())
                listed_testing.append(vectorized_testing)

            train_vectors1 = np.asarray(listed1)
            train_vectors2 = []
            test_vectors = np.asarray(listed_testing)

            print('average vectors')

            train_vectors = data_processing(scan_count, train_vectors1, train_vectors2)

            # #############################################################################
            # Import coordinates for fixations

            print('importing fixations')

            fixations = pd.read_csv('stim_vals.csv')
            x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1)*monitor_width/2, scan_count-1)
            y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1)*monitor_height/2, scan_count-1)

            # #############################################################################
            # Create SVR Model

            predicted_x, predicted_y = create_model(train_vectors, test_vectors, x_targets, y_targets)

            # #############################################################################
            # Plot SVR predictions against targets

            # scatter_plot(name, x_targets, y_targets, predicted_x, predicted_y, plot=True, save=False)
            axis_plot(fixations, predicted_x, predicted_y)

            print('Completed participant ' + name)

            # ###############################################################################
            # Get error measurements

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

            print([x_error, y_error])

            params.loc[name, 'x_error_' + str(scan)] = x_error
            params.loc[name, 'y_error_' + str(scan)] = y_error

            try:

                fd1 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_1']['RMSD (Mean)'])
                fd2 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_2']['RMSD (Mean)'])
                dv1 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_1']['Std. DVARS (Mean)'])
                dv2 = float(qap[qap['Participant'] == name][qap['Series'] == 'func_peer_run_2']['Std. DVARS (Mean)'])

                fd_score = np.average([[fd1, fd2]])
                dvars_score = np.average([dv1, dv2])
                params.loc[name, 'mean_fd'] = fd_score
                params.loc[name, 'dvars'] = dvars_score

            except:

                print('Participant not found in QAP')

            if update:

                params.to_csv('subj_params.csv')

        except:

            print('Error processing participant')

# icc_peer(full_set, gsr=True, update=True, scan=1)
icc_peer(['sub-5437909'], gsr=True, update=False, scan=3)

params = pd.read_csv('subj_params.csv', index_col='subject', dtype=object)
params = params[params['scan_count'] == str(3)]
p1x = params['x_error_1']
p1y = params['y_error_1']
p3x = params['x_error_3']
p3y = params['y_error_3']
ptx = params['x_error_gsr']
pty = params['y_error_gsr']
p1x = np.array([float(x) for x in p1x])
p1y = np.array([float(x) for x in p1y])
p3x = np.array([float(x) for x in p3x])
p3y = np.array([float(x) for x in p3y])
ptx = np.array([float(x) for x in ptx])
pty = np.array([float(x) for x in pty])

def compute_icc(x, y):
    """
    This function computes the inter-class correlation (ICC) of the
    two classes represented by the x and y numpy vectors.
    """
    n = len(x)

    if all(x == y):
        return 1

    Sx = sum(x); Sy = sum(y);
    Sxx = sum(x*x); Sxy = sum( (x+y)**2 )/2; Syy = sum(y*y)

    fact = ((Sx + Sy)**2)/(n*2)
    SS_tot = Sxx + Syy - fact
    SS_among = Sxy - fact
    SS_error = SS_tot - SS_among

    MS_error = SS_error/n
    MS_among = SS_among/(n-1)

    ICC = (MS_among - MS_error) / (MS_among + MS_error)

    return ICC

compute_icc(p3x, ptx)

from scipy.stats import pearsonr

pearsonr(p1y, p3y)

p1_ptx = ttest_rel(p1x, ptx)[1]
p1_p3x = ttest_rel(p1x, p3x)[1]
p1_pty = ttest_rel(p1y, pty)[1]
p1_p3y = ttest_rel(p1y, p3y)[1]

print([p1_ptx, p1_p3x, p1_pty, p1_p3y])

# #############################################################################
# Test registered data

eye_mask = nib.load('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_eye_mask.nii.gz')
eye_mask = eye_mask.get_data()

params = pd.read_csv('subj_params.csv', index_col='subject', dtype=object)
sub_ref = params.index.values.tolist()

reg_list = []

with open('subj_params.csv', 'a') as updated_params:
    writer = csv.writer(updated_params)

    for subject in os.listdir('/data2/Projects/Jake/Registration_complete/'):
        if any(subject in x for x in sub_ref) and 'txt' not in subject:
            if str(params.loc[subject, 'x_error_reg']) == 'nan':
                reg_list.append(subject)

# params = pd.read_csv('subj_params.csv', index_col='subject')
# output = params[params['x_error_gsr'] < 350][params['y_error_gsr'] < 350][params['scan_count'] == 3]

def regi_peer(reg_list):


    for sub in reg_list:

        print('starting participant ' + str(sub))
        scan1 = nib.load(resample_path + sub + '/peer1_eyes.nii.gz')
        scan1 = scan1.get_data()
        scan2 = nib.load(resample_path + sub + '/peer2_eyes.nii.gz')
        scan2 = scan2.get_data()
        scan3 = nib.load(resample_path + sub + '/peer3_eyes.nii.gz')
        scan3 = scan3.get_data()

        for item in [scan1, scan2, scan3]:

            for vol in range(item.shape[3]):

                output = np.multiply(eye_mask, item[:, :, :, vol])

                item[:, :, :, vol] = output

        for item in [scan1, scan2, scan3]:

            item = mean_center_var_norm(item)
            item = gs_regress(item, 0, item.shape[0] - 1, 0, item.shape[1] - 1, 0, item.shape[2] - 1)

        listed1 = []
        listed2 = []
        listed_testing = []

        print('beginning vectors')

        for tr in range(int(scan1.shape[3])):

            tr_data1 = scan1[:, :, :, tr]
            vectorized1 = np.array(tr_data1.ravel())
            listed1.append(vectorized1)

            tr_data2 = scan3[:, :, :, tr]
            vectorized2 = np.array(tr_data2.ravel())
            listed2.append(vectorized2)

            te_data = scan2[:, :, :, tr]
            vectorized_testing = np.array(te_data.ravel())
            listed_testing.append(vectorized_testing)

        train_vectors1 = np.asarray(listed1)
        test_vectors = np.asarray(listed_testing)
        train_vectors2 = np.asarray(listed2)

        # #############################################################################
        # Averaging training signal

        print('average vectors')

        train_vectors = data_processing(3, train_vectors1, train_vectors2)

        # #############################################################################
        # Import coordinates for fixations

        print('importing fixations')

        fixations = pd.read_csv('stim_vals.csv')
        x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1) * monitor_width / 2, 3 - 1)
        y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1) * monitor_height / 2, 3 - 1)

        # #############################################################################
        # Create SVR Model

        x_model, y_model = create_model(train_vectors, x_targets, y_targets)

        predicted_x, predicted_y = predict_fixations(x_model, y_model, test_vectors)

        axis_plot(fixations, predicted_x, predicted_y, sub, train_sets=1)

        x_res = []
        y_res = []

        for num in range(27):

            nums = num * 5

            for values in range(5):
                error_x = (abs(x_targets[num] - predicted_x[nums + values])) ** 2
                error_y = (abs(y_targets[num] - predicted_y[nums + values])) ** 2
                x_res.append(error_x)
                y_res.append(error_y)

        x_error = np.sqrt(np.sum(np.array(x_res)) / 135)
        y_error = np.sqrt(np.sum(np.array(y_res)) / 135)
        print([x_error, y_error])

        params.loc[sub, 'x_error_reg'] = x_error
        params.loc[sub, 'y_error_reg'] = y_error
        params.to_csv('subj_params.csv')
        print('participant ' + str(sub) + ' complete')

full_list = os.listdir('/data2/Projects/Jake/Registration_complete/')

ants_data = '/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/output/pipeline_RU_CPAC/sub-5002891_ses-1/motion_correct_to_standard/'

x_non_reg = params.loc[full_list, 'x_error_gsr'].tolist()
x_with_reg = params.loc[full_list, 'x_error_reg'].tolist()
y_non_reg = params.loc[full_list, 'y_error_gsr'].tolist()
y_with_reg = params.loc[full_list, 'y_error_reg'].tolist()

x_non_reg = np.array([float(x) for x in x_non_reg])
x_with_reg = np.array([float(x) for x in x_with_reg])
y_non_reg = np.array([float(x) for x in y_non_reg])
y_with_reg = np.array([float(x) for x in y_with_reg])

m1, b1 = np.polyfit(x_non_reg, x_with_reg, 1)
m2, b2 = np.polyfit(y_non_reg, y_with_reg, 1)

plt.figure(figsize=(8, 8))

plt.scatter(x_non_reg, x_with_reg, color='b')
plt.scatter(y_non_reg, y_with_reg, color='r')
plt.plot(x_non_reg, m1*x_non_reg + b1, color='b', label='x, slope=.56')
plt.plot(y_non_reg, m2*y_non_reg + b2, color='r', label='y, slope=.52')
plt.xlim([0, 1400])
plt.ylim([0, 1400])
plt.xlabel('RMSE without registration')
plt.ylabel('RMSE with registration')
plt.legend()

plt.show()

# #############################################################################
# Generalizable classifier

cpac_path = '/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/output/pipeline_RU_CPAC/'

# eye_mask = nib.load('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_eye_mask.nii.gz')
eye_mask = nib.load('/data2/Projects/Jake/eye_eroded.nii.gz')
eye_mask = eye_mask.get_data()

params = pd.read_csv('subj_params.csv', index_col='subject', dtype=object)
sub_ref = params.index.values.tolist()

reg_list = []

params = pd.read_csv('subj_params.csv', index_col='subject')

with open('subj_params.csv', 'a') as updated_params:
    writer = csv.writer(updated_params)

    for subject in os.listdir(cpac_path):

        subject = subject.replace('_ses-1', '')
        try:
            if int(params.loc[subject, 'scan_count']) == 3:
                if float(params.loc[subject, 'x_error_gsr']) < 400:
                    if float(params.loc[subject, 'y_error_gsr']) < 400:
                        reg_list.append(subject)
        except:
            continue

reg_list = reg_list[:50]

# with open('subj_params.csv', 'a') as updated_params:
#     writer = csv.writer(updated_params)
#
#     for subject in os.listdir(cpac_path):
#         if any(subject in x for x in sub_ref) and 'txt' not in subject:
#             if str(params.loc[subject, 'x_error_reg']) != 'nan':
#                 reg_list.append(subject)

eye_mask = nib.load('/data2/Projects/Jake/Resampled/eye_all_sub.nii.gz')
eye_mask = eye_mask.get_data()

reg_list = ['sub-5161675', 'sub-5169146', 'sub-5343770', 'sub-5375858', 'sub-5581172', 'sub-5629350',
            'sub-5637071', 'sub-5797959', 'sub-5930252', 'sub-5974505']

reg_list = ['sub-5438434', 'sub-5171285', 'sub-5909780', 'sub-5637071', 'sub-5292617', 'sub-5917648',
            'sub-5665223', 'sub-5375858', 'sub-5862879', 'sub-5124198', 'sub-5072464', 'sub-5469524',
            'sub-5385307', 'sub-5271530', 'sub-5481682', 'sub-5905922', 'sub-5773707', 'sub-5745590',
            'sub-5185233', 'sub-5696548', 'sub-5054883', 'sub-5484500', 'sub-5171529', 'sub-5340375',
            'sub-5270411', 'sub-5378545', 'sub-5032610', 'sub-5310335', 'sub-5984037', 'sub-5814325',
            'sub-5169146', 'sub-5289010', 'sub-5351657', 'sub-5707321', 'sub-5604492', 'sub-5974505',
            'sub-5307785', 'sub-5303849', 'sub-5986705', 'sub-5787700', 'sub-5659524', 'sub-5844932',
            'sub-5263388', 'sub-5397290', 'sub-5161062', 'sub-5797339', 'sub-5975698', 'sub-5260373',
            'sub-5276304']

reg_list = ['sub-5002891', 'sub-5005437']

reg_list = ['sub-5986705','sub-5375858','sub-5292617','sub-5397290','sub-5844932','sub-5787700','sub-5797959',
            'sub-5378545','sub-5085726','sub-5984037','sub-5076391','sub-5263388','sub-5171285',
            'sub-5917648','sub-5814325','sub-5169146','sub-5484500','sub-5481682','sub-5232535','sub-5905922',
            'sub-5975698','sub-5986705','sub-5343770']

train_set_count = len(reg_list) - 1

resample_path = '/data2/Projects/Jake/Resampled/'


def general_classifier(reg_list):

    funcTime = datetime.now()

    train_vectors1 = []
    train_vectors2 = []
    test_vectors = []

    for sub in reg_list[:train_set_count]:

        print('starting participant ' + str(sub))
        # scan1 = nib.load(cpac_path + sub + '_ses-1/motion_correct_to_standard/_scan_peer_run-1/' + sub + '_task-peer_run-1_bold_calc_tshift_resample_volreg_antswarp.nii.gz')
        # scan1 = scan1.get_data()
        # scan2 = nib.load(cpac_path + sub + '_ses-1/motion_correct_to_standard/_scan_peer_run-2/' + sub + '_task-peer_run-2_bold_calc_tshift_resample_volreg_antswarp.nii.gz')
        # scan2 = scan2.get_data()
        # scan3 = nib.load(cpac_path + sub + '_ses-1/motion_correct_to_standard/_scan_peer_run-3/' + sub + '_task-peer_run-3_bold_calc_tshift_resample_volreg_antswarp.nii.gz')
        # scan3 = scan3.get_data()

        scan1 = nib.load(resample_path + sub + '/peer1_eyes.nii.gz')
        scan1 = scan1.get_data()
        scan2 = nib.load(resample_path + sub + '/peer2_eyes.nii.gz')
        scan2 = scan2.get_data()
        scan3 = nib.load(resample_path + sub + '/peer3_eyes.nii.gz')
        scan3 = scan3.get_data()

        for item in [scan1, scan2, scan3]:

            for vol in range(item.shape[3]):

                output = np.multiply(eye_mask, item[:, :, :, vol])

                item[:, :, :, vol] = output

        for item in [scan1, scan2, scan3]:

            item = mean_center_var_norm(item)
            item = gs_regress(item, 0, item.shape[0]-1, 0, item.shape[1]-1, 0, item.shape[2]-1)

        listed1 = []
        listed2 = []
        listed_testing = []

        print('beginning vectors')

        for tr in range(int(scan1.shape[3])):

            tr_data1 = scan1[:,:,:, tr]
            vectorized1 = np.array(tr_data1.ravel())
            listed1.append(vectorized1)

            tr_data2 = scan3[:,:,:, tr]
            vectorized2 = np.array(tr_data2.ravel())
            listed2.append(vectorized2)

            te_data = scan2[:,:,:, tr]
            vectorized_testing = np.array(te_data.ravel())
            listed_testing.append(vectorized_testing)

        train_vectors1.append(listed1)
        test_vectors.append(listed_testing)
        train_vectors2.append(listed2)

    full_train1 = []
    full_test = []
    full_train2 = []

    for part in range(len(reg_list[:train_set_count])):
        for vect in range(scan1.shape[3]):
            full_train1.append(train_vectors1[part][vect])
            full_test.append(test_vectors[part][vect])
            full_train2.append(train_vectors2[part][vect])

        # train_vectors1 = np.asarray(listed1)
        # test_vectors = np.asarray(listed_testing)
        # train_vectors2 = np.asarray(listed2)

        # #############################################################################
        # Averaging training signal

    print('average vectors')

    train_vectors = data_processing(3, full_train1, full_train2)

        # #############################################################################
        # Import coordinates for fixations

    print('importing fixations')

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), len(reg_list[:train_set_count])) * monitor_width / 2, 3 - 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), len(reg_list[:train_set_count])) * monitor_height / 2, 3 - 1)

        # #############################################################################
        # Create SVR Model

    x_model, y_model = create_model(train_vectors, x_targets, y_targets)
    print('Training completed: ' + str(datetime.now() - funcTime))

    for gen in range(len(reg_list)):

    # gen = 1
        gen = gen+1
        predicted_x = x_model.predict(full_test[scan1.shape[3]*(gen-1):scan1.shape[3]*(gen)])
        predicted_y = y_model.predict(full_test[scan1.shape[3]*(gen-1):scan1.shape[3]*(gen)])
        axis_plot(fixations, predicted_x, predicted_y, sub, train_sets=1)


        x_res = []
        y_res = []

        sub = reg_list[gen-1]

        for num in range(27):

            nums = num * 5

            for values in range(5):
                error_x = (abs(x_targets[num] - predicted_x[nums + values])) ** 2
                error_y = (abs(y_targets[num] - predicted_y[nums + values])) ** 2
                x_res.append(error_x)
                y_res.append(error_y)

        x_error = np.sqrt(np.sum(np.array(x_res)) / 135)
        y_error = np.sqrt(np.sum(np.array(y_res)) / 135)
        print([x_error, y_error])

        params.loc[sub, 'x_error_within'] = x_error
        params.loc[sub, 'y_error_within'] = y_error
        params.to_csv('subj_params.csv')
        print('participant ' + str(sub) + ' complete')


x_targets = np.repeat(np.array(fixations['pos_x']), 5 * 1) * monitor_width / 2
y_targets = np.repeat(np.array(fixations['pos_y']), 5 * 1) * monitor_height / 2

time_series = range(0, len(p1))

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.ylabel('Horizontal position')
plt.plot(time_series, x_targets, '.-', color='k', label='Fixations')
plt.plot(time_series, p1, '.-', color='b', label='General')
plt.plot(time_series, p3, '.-', color='r', label='Subject')
plt.legend()
plt.subplot(2, 1, 2)
plt.ylabel('Vertical position')
plt.xlabel('TR')
plt.plot(time_series, y_targets, '.-', color='k', label='Fixations')
plt.plot(time_series, p2, '.-', color='b', label='General')
plt.plot(time_series, p4, '.-', color='r', label='Subject')
plt.legend()
plt.show()

# #############################################################################
# Visualize error vs motion

# params = pd.read_csv('subj_params.csv', index_col='subject')
# params = params[params['x_error'] < 50000][params['y_error'] < 50000][params['mean_fd'] < 3.8][params['dvars'] < 1.5]
#
# # Need to fix script to not rely on indexing and instead include a subset based on mean and stdv parameters
# num_part = len(params)
#
# x_error_list = params.loc[:, 'x_error'][:num_part].tolist()
# y_error_list = params.loc[:, 'y_error'][:num_part].tolist()
# mean_fd_list = params.loc[:, 'mean_fd'][:num_part].tolist()
# dvars_list = params.loc[:, 'dvars'][:num_part].tolist()
#
# x_error_list = np.array([float(x) for x in x_error_list])
# y_error_list = np.array([float(x) for x in y_error_list])
# mean_fd_list = np.array([float(x) for x in mean_fd_list])
# dvars_list = np.array([float(x) for x in dvars_list])
#
# m1, b1 = np.polyfit(mean_fd_list, x_error_list, 1)
# m2, b2 = np.polyfit(mean_fd_list, y_error_list, 1)
# m3, b3 = np.polyfit(dvars_list, x_error_list, 1)
# m4, b4 = np.polyfit(dvars_list, y_error_list, 1)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 2, 1)
# plt.title('mean_fd vs. x_RMS')
# plt.scatter(mean_fd_list, x_error_list, s=5)
# plt.plot(mean_fd_list, m1*mean_fd_list + b1, '-', color='r')
# plt.subplot(2, 2, 2)
# plt.title('mean_fd vs. y_RMS')
# plt.scatter(mean_fd_list, y_error_list, s=5)
# plt.plot(mean_fd_list, m2*mean_fd_list + b2, '-', color='r')
# plt.subplot(2, 2, 3)
# plt.title('dvars vs. x_RMS')
# plt.scatter(dvars_list, x_error_list, s=5)
# plt.plot(dvars_list, m3*dvars_list + b3, '-', color='r')
# plt.subplot(2, 2, 4)
# plt.title('dvars vs. y_RMS')
# plt.scatter(dvars_list, y_error_list, s=5)
# plt.plot(dvars_list, m4*dvars_list + b4, '-', color='r')
# plt.show()
