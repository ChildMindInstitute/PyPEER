import os
import csv
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVR
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from aux_process import *
from joblib import Parallel, delayed
import pickle
import seaborn as sns
import matplotlib.ticker as ticker

monitor_width = 1680
monitor_height = 1050

# #############################################################################
# Update subject parameters sheet with new participants

# data_path = '/data2/Projects/Jake/Human_Brain_Mapping/'
# qap_path_RU = '/data2/HBNcore/CMI_HBN_Data/MRI/RU/QAP/qap_functional_temporal.csv'
# qap_path_CBIC = '/data2/HBNcore/CMI_HBN_Data/MRI/CBIC/QAP/qap_functional_temporal.csv'
# output_path = '/home/json/Desktop/peer/Figures'

# #############################################################################
# Import data


def standard_peer(subject_list, gsr=True, update=False):

    for name in subject_list:

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

                    training1_data = gs_regress(training1_data)
                    testing_data = gs_regress(testing_data)

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

# #############################################################################
# Test registered data

# eye_mask = nib.load('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_eye_mask.nii.gz')
eye_mask = nib.load('/data2/Projects/Jake/eye_masks/2mm_eye_corrected.nii.gz')
eye_mask = eye_mask.get_data()
resample_path = '/data2/Projects/Jake/Human_Brain_Mapping/'

def regi_peer(reg_list):

    for sub in reg_list:

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)

        print('starting participant ' + str(sub))

        scan_count = int(params.loc[sub, 'scan_count'])

        try:

            viewtype = 'calibration'

            if viewtype == 'calibration':
                second_file = '/peer2_eyes_sub.nii.gz'
            elif viewtype == 'tp':
                second_file = '/movie_TP_eyes_sub.nii.gz'
            elif viewtype == 'dm':
                second_file = '/movie_D_eyes_sub.niigz'

            scan1 = nib.load(resample_path + sub + '/peer1_eyes_sub.nii.gz')
            scan1 = scan1.get_data()
            print('Scan 1 loaded')
            scan2 = nib.load(resample_path + sub + second_file)
            scan2 = scan2.get_data()
            print('Scan 2 loaded')
            scan3 = nib.load(resample_path + sub + '/peer3_eyes_sub.nii.gz')
            scan3 = scan3.get_data()
            print('Scan 3 loaded')

            print('Applying eye-mask')

            for item in [scan1, scan2, scan3]:

                for vol in range(item.shape[3]):

                    output = np.multiply(eye_mask, item[:, :, :, vol])

                    item[:, :, :, vol] = output

            print('Applying mean-centering with variance-normalization and GSR')

            for item in [scan1, scan2, scan3]:

                item = mean_center_var_norm(item)
                item = gs_regress(item, eye_mask)

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

            for tr in range(int(scan2.shape[3])):

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
            predicted_x = np.array([np.round(float(x), 3) for x in predicted_x])
            predicted_y = np.array([np.round(float(x), 3) for x in predicted_y])

            # x_targets, y_targets = axis_plot(fixations, predicted_x, predicted_y, sub, train_sets=1)
            # movie_plot(predicted_x, predicted_y, sub, train_sets=1)

            x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
            y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

            x_corr = compute_icc(predicted_x, x_targets)
            y_corr = compute_icc(predicted_y, y_targets)

            x_error_sk = np.sqrt(mean_squared_error(predicted_x, x_targets))
            y_error_sk = np.sqrt(mean_squared_error(predicted_y, y_targets))

            params.loc[sub, 'rmse_x_gsr'] = x_error_sk
            params.loc[sub, 'rmse_y_gsr'] = y_error_sk
            params.loc[sub, 'corr_x_gsr'] = x_corr
            params.loc[sub, 'corr_y_gsr'] = y_corr
            params.loc[sub, 'pred_x_gsr'] = predicted_x
            params.loc[sub, 'pred_y_gsr'] = predicted_y
            params.loc[sub, 'feat_x_gsr'] = [x for x in x_model.coef_[0]]
            params.loc[sub, 'feat_y_gsr'] = [x for x in y_model.coef_[0]]
            params.to_csv('model_outputs.csv')
            print('participant ' + str(sub) + ' complete')

        except:
            continue


def three_valid(sub, gsr_, second_file, viewtype):

    if viewtype == 'calibration':

        print('For analysis of subjects with three calibration scans')

        scan1 = nib.load(resample_path + sub + '/peer1_eyes_sub.nii.gz')
        scan1 = scan1.get_data()
        print('Scan 1 loaded')
        scan2 = nib.load(resample_path + sub + second_file)
        scan2 = scan2.get_data()
        print('Scan 2 loaded')
        scan3 = nib.load(resample_path + sub + '/peer3_eyes_sub.nii.gz')
        scan3 = scan3.get_data()
        print('Scan 3 loaded')

        print('Applying eye-mask')

        for item in [scan1, scan2, scan3]:

            for vol in range(item.shape[3]):
                output = np.multiply(eye_mask, item[:, :, :, vol])

                item[:, :, :, vol] = output

        print('Applying mean-centering with variance-normalization and GSR')

        for item in [scan1, scan2, scan3]:
            item = mean_center_var_norm(item)

            if int(gsr_) == 1:

                item = gs_regress(item, eye_mask)

            else:

                continue

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

        for tr in range(int(scan2.shape[3])):
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

    else:

        # What needs to be changed to have all necessary parameters to predict? x_model, y_model, test_vectors

        scan2 = nib.load(resample_path + sub + second_file)
        scan2 = scan2.get_data()
        print('Scan 2 loaded')

        print('Applying eye-mask')

        for item in [scan2]:

            for vol in range(item.shape[3]):
                output = np.multiply(eye_mask, item[:, :, :, vol])

                item[:, :, :, vol] = output

        print('Applying mean-centering with variance-normalization and GSR')

        for item in [scan2]:
            item = mean_center_var_norm(item)

            if int(gsr_) == 1:

                item = gs_regress(item, eye_mask)

            else:

                continue

        listed1 = []
        listed2 = []
        listed_testing = []

        print('beginning vectors')

        for tr in range(int(scan2.shape[3])):
            te_data = scan2[:, :, :, tr]
            vectorized_testing = np.array(te_data.ravel())
            listed_testing.append(vectorized_testing)

        test_vectors = np.asarray(listed_testing)

        print('importing fixations')

        fixations = pd.read_csv('stim_vals.csv')
        x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1) * monitor_width / 2, 2 - 1)
        y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1) * monitor_height / 2, 2 - 1)

        x_model = pickle.load(open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/x_gsr_model.sav', 'rb'))
        y_model = pickle.load(open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/y_gsr_model.sav', 'rb'))

    predicted_x, predicted_y = predict_fixations(x_model, y_model, test_vectors)
    predicted_x = np.array([np.round(float(x), 3) for x in predicted_x])
    predicted_y = np.array([np.round(float(x), 3) for x in predicted_y])

    # x_targets, y_targets = axis_plot(fixations, predicted_x, predicted_y, sub, train_sets=1)
    # movie_plot(predicted_x, predicted_y, sub, train_sets=1)

    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

    if viewtype == 'calibration':

        x_corr = compute_icc(predicted_x, x_targets)
        y_corr = compute_icc(predicted_y, y_targets)

        x_error_sk = np.sqrt(mean_squared_error(predicted_x, x_targets))
        y_error_sk = np.sqrt(mean_squared_error(predicted_y, y_targets))

    else:

        x_corr = 0; y_corr = 0; x_error_sk = 0; y_error_sk = 0

    return x_error_sk, y_error_sk, x_corr, y_corr, predicted_x, predicted_y, x_model, y_model


def two_valid(sub, gsr_, second_file, viewtype):

    print('For analysis of subjects with two calibration scans')

    if viewtype == 'calibration':

        scan1 = nib.load(resample_path + sub + '/peer1_eyes_sub.nii.gz')
        scan1 = scan1.get_data()
        print('Scan 1 loaded')
        scan2 = nib.load(resample_path + sub + second_file)
        scan2 = scan2.get_data()
        print('Scan 2 loaded')

        print('Applying eye-mask')

        for item in [scan1, scan2]:

            for vol in range(item.shape[3]):
                output = np.multiply(eye_mask, item[:, :, :, vol])

                item[:, :, :, vol] = output

        print('Applying mean-centering with variance-normalization and GSR')

        for item in [scan1, scan2]:
            item = mean_center_var_norm(item)

            if int(gsr_) == 1:

                item = gs_regress(item, eye_mask)

            else:

                continue

        listed1 = []
        listed2 = []
        listed_testing = []

        print('beginning vectors')

        for tr in range(int(scan1.shape[3])):
            tr_data1 = scan1[:, :, :, tr]
            vectorized1 = np.array(tr_data1.ravel())
            listed1.append(vectorized1)

        for tr in range(int(scan2.shape[3])):
            te_data = scan2[:, :, :, tr]
            vectorized_testing = np.array(te_data.ravel())
            listed_testing.append(vectorized_testing)

        train_vectors1 = np.asarray(listed1)
        test_vectors = np.asarray(listed_testing)

        # #############################################################################
        # Averaging training signal

        print('average vectors')

        train_vectors2 = []

        train_vectors = data_processing(2, train_vectors1, train_vectors2)

        # #############################################################################
        # Import coordinates for fixations

        print('importing fixations')

        fixations = pd.read_csv('stim_vals.csv')
        x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1) * monitor_width / 2, 2 - 1)
        y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1) * monitor_height / 2, 2 - 1)

        # #############################################################################
        # Create SVR Model

        x_model, y_model = create_model(train_vectors, x_targets, y_targets)

    else:

        # What needs to be changed to have all necessary parameters to predict? x_model, y_model, test_vectors

        scan2 = nib.load(resample_path + sub + second_file)
        scan2 = scan2.get_data()
        print('Scan 2 loaded')

        print('Applying eye-mask')

        for item in [scan2]:

            for vol in range(item.shape[3]):
                output = np.multiply(eye_mask, item[:, :, :, vol])

                item[:, :, :, vol] = output

        print('Applying mean-centering with variance-normalization and GSR')

        for item in [scan2]:
            item = mean_center_var_norm(item)

            if int(gsr_) == 1:

                item = gs_regress(item, eye_mask)

            else:

                continue

        listed1 = []
        listed2 = []
        listed_testing = []

        print('beginning vectors')

        for tr in range(int(scan2.shape[3])):
            te_data = scan2[:, :, :, tr]
            vectorized_testing = np.array(te_data.ravel())
            listed_testing.append(vectorized_testing)

        test_vectors = np.asarray(listed_testing)

        print('importing fixations')

        fixations = pd.read_csv('stim_vals.csv')
        x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1) * monitor_width / 2, 2 - 1)
        y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1) * monitor_height / 2, 2 - 1)

        x_model = pickle.load('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/x_gsr_model.sav', 'rb')
        y_model = pickle.load('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/y_gsr_model.sav', 'rb')

    predicted_x, predicted_y = predict_fixations(x_model, y_model, test_vectors)
    predicted_x = np.array([np.round(float(x), 3) for x in predicted_x])
    predicted_y = np.array([np.round(float(x), 3) for x in predicted_y])

    # x_targets, y_targets = axis_plot(fixations, predicted_x, predicted_y, sub, train_sets=1)
    # movie_plot(predicted_x, predicted_y, sub, train_sets=1)

    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

    if viewtype == 'calibration':

        x_corr = compute_icc(predicted_x, x_targets)
        y_corr = compute_icc(predicted_y, y_targets)

        x_error_sk = np.sqrt(mean_squared_error(predicted_x, x_targets))
        y_error_sk = np.sqrt(mean_squared_error(predicted_y, y_targets))

    else:

        x_corr = 0; y_corr = 0; x_error_sk = 0; y_error_sk = 0

    return x_error_sk, y_error_sk, x_corr, y_corr, predicted_x, predicted_y, x_model, y_model


def update_output(params, gsr_, sub, x_error_sk, y_error_sk, x_corr, y_corr, predicted_x, predicted_y, x_model, y_model, save_name, param_name, viewtype):

    print('Updating output for subject ' + str(sub))

    param_dict = {'sub': [sub, sub],'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []}
    output_dict = {'x_pred': [], 'y_pred': []}

    if viewtype == 'calibration':

        param_dict['corr_x'] = x_corr
        param_dict['corr_y'] = y_corr
        param_dict['rmse_x'] = x_error_sk
        param_dict['rmse_y'] = y_error_sk
        output_dict['x_pred'] = predicted_x
        output_dict['y_pred'] = predicted_y

        if gsr_ == 1:

            df_p = pd.DataFrame(param_dict)
            df_p.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/parameters_gsr.csv')
            df_o = pd.DataFrame(output_dict)
            df_o.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/predictions_gsr.csv')

            pickle.dump(x_model, open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/x_gsr_model.sav', 'wb'))
            pickle.dump(y_model, open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/y_gsr_model.sav', 'wb'))

        else:

            df_p = pd.DataFrame(param_dict)
            df_p.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/parameters_no_gsr.csv')
            df_o = pd.DataFrame(output_dict)
            df_o.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/predictions_no_gsr.csv')

            pickle.dump(x_model, open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/x_no_gsr_model.sav', 'wb'))
            pickle.dump(y_model, open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/y_no_gsr_model.sav', 'wb'))

    else:

        output_dict['x_pred'] = predicted_x
        output_dict['y_pred'] = predicted_y

        df_o = pd.DataFrame(output_dict)
        df_o.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/' + str(viewtype) + 'predictions.csv')

    print('participant ' + str(sub) + ' complete')


def peer_hbm(sub, viewtype='calibration', gsr_='on'):

    print('Starting with participant ' + str(sub) + ' for viewing ' + str(viewtype) + ' with gsr ' + str(gsr_))

    try:

        if gsr_ == 'on':
            gsr_ = 1
        else:
            gsr_ = 0

        fixations = pd.read_csv('stim_vals.csv')
        x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1) * monitor_width / 2, 3 - 1)
        y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1) * monitor_height / 2, 3 - 1)

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)

        scan_count = int(params.loc[sub, 'scan_count'])

        param_name = []

        if (viewtype == 'calibration') & (gsr_ == 1):
            second_file = '/peer2_eyes_sub.nii.gz'
            param_name = 'gsr_params.csv'
            save_name = 'gsr_pred.csv'
        elif (viewtype == 'calibration') & (gsr_ == 0):
            second_file = '/peer2_eyes_sub.nii.gz'
            param_name = 'no_gsr_params.csv'
            save_name = 'no_gsr_pred.csv'
        elif viewtype == 'tp':
            second_file = '/movie_TP_eyes_sub.nii.gz'
            save_name = 'tp_pred.csv'
        elif viewtype == 'dm':
            second_file = '/movie_DM_eyes_sub.nii.gz'
            save_name = 'dm_pred.csv'

        if scan_count == 3:

            x_error_sk, y_error_sk, x_corr, y_corr, predicted_x, predicted_y, x_model, y_model = three_valid(sub, gsr_, second_file, viewtype)
            update_output(params, gsr_, sub, x_error_sk, y_error_sk, x_corr, y_corr, predicted_x, predicted_y, x_model, y_model, save_name, param_name, viewtype)

        elif scan_count == 2:

            x_error_sk, y_error_sk, x_corr, y_corr, predicted_x, predicted_y, x_model, y_model = two_valid(sub, gsr_, second_file, viewtype)
            update_output(params, gsr_, sub, x_error_sk, y_error_sk, x_corr, y_corr, predicted_x, predicted_y, x_model, y_model, save_name, param_name, viewtype)

        else:

            print('Not enough scans for analysis')

    except:

        print('Error processing subject ' + str(sub))

    # params.to_csv('model_outputs.csv')

params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
sub_list = params.index.values.tolist()

sub_list = []

for sub in os.listdir(resample_path):
    if os.path.exists(resample_path + str(sub) + '/parameters_gsr.csv'):
        continue
    else:
        sub_list.append(sub)

sub_list = ['sub-5157882']

Parallel(n_jobs=10)(delayed(peer_hbm)(sub, viewtype='calibration', gsr_='on')for sub in sub_list)

def pred_aggregate(gsr_status='on', viewtype='calibration', motion_type='mean_fd'):

    if gsr_status == 'off':
        filename = 'predictions_no_gsr.csv'
        outname = viewtype + '_no_gsr_' + motion_type
    elif gsr_status == 'on':
        if viewtype == 'calibration':
            filename = 'predictions_gsr.csv'
            outname = viewtype + '_gsr_' + motion_type
        elif viewtype == 'tp':
            filename = 'tppredictions.csv'
            outname = 'tp_' + motion_type
        elif viewtype == 'dm':
            filename = 'dmpredictions.csv'
            outname = 'dm_' + motion_type

    outname_x = outname + str('_x')
    outname_y = outname + str('_y')

    monitor_width = 1680
    monitor_height = 1050

    params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
    params = params.convert_objects(convert_numeric=True)
    params = params[(params.scan_count == 3) | (params.scan_count == 2)]
    params = params.sort_values(by=motion_type, ascending=True)

    sub_list = params.index.values.tolist()

    x_temp = []
    y_temp = []
    count = 0

    for sub in sub_list:

        try:

            if count < len(sub_list):

                data = pd.read_csv(resample_path + str(sub) + '/'+ filename)
                x_out = list(data['x_pred'])
                y_out = list(data['y_pred'])

                if count == 0:
                    expected_len = len(x_out)

                if expected_len == len(x_out):

                    for x in range(len(x_out)):
                        if abs(x_out[x]) > monitor_width/2 + .10 * monitor_width:
                            x_out[x] = 0
                        else:
                            x_out[x] = x_out[x]

                    for x in range(len(y_out)):
                        if abs(y_out[x]) > monitor_height/2 + .10 * monitor_height:
                            y_out[x] = 0
                        else:
                            y_out[x] = y_out[x]

                    x_out = np.array(x_out)
                    y_out = np.array(y_out)

                    x_temp.append(x_out)
                    y_temp.append(y_out)

                count += 1

            else:

                break

        except:

            continue

    # Import fixations

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_width / 2, 1)

    arr = np.zeros(len(x_out))
    arrx = np.array([-np.round(monitor_width/2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height/2, 0) for x in arr])

    if viewtype == 'calibration':

        for num in range(int(np.round(len(sub_list)*.02, 0))):
            x_temp.append(arrx)
            y_temp.append(arry)

        for num in range(int(np.round(len(sub_list)*.02, 0))):
            x_temp.append(x_targets)
            y_temp.append(y_targets)

    else:

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_temp.append(arrx)
            y_temp.append(arry)

        for num in range(int(np.round(len(sub_list) * .02, 0))):
            x_temp.append(np.mean(x_temp, axis=0))
            y_temp.append(np.mean(y_temp, axis=0))

    x_hm = np.stack(x_temp)
    y_hm = np.stack(y_temp)

    save_heatmap(x_hm, outname_x)
    save_heatmap(y_hm, outname_y)

    return x_hm, y_hm

x_hm, y_hm = pred_aggregate(gsr_status='off', viewtype='calibration', motion_type='mean_fd')

for views in ['calibration', 'tp', 'dm']:
    for motions in ['mean_fd', 'dvars']:
        x_hm, y_hm = pred_aggregate(gsr_status = 'on', viewtype=views, motion_type=motions)

for motions in ['mean_fd', 'dvars']:
    x_hm, y_hm = pred_aggregate(gsr_status='off', viewtype='calibration', motion_type=motions)

def save_heatmap(model, outname):

    sns.set()
    plt.clf()
    ax = sns.heatmap(model)
    ax.set(xlabel='Volumes', ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
    plt.savefig('/home/json/Desktop/peer/hbm_figures/' + outname + '.png', dpi=600)
    plt.show()

g = sns.clustermap(x_hm, col_cluster=False)

def create_swarms():

    g_corr_x = []
    g_corr_y = []
    n_corr_x = []
    n_corr_y = []
    g_rmse_x = []
    g_rmse_y = []
    n_rmse_x = []
    n_rmse_y = []

    params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
    params = params.convert_objects(convert_numeric=True)
    params = params[params.scan_count == 3]
    sub_list = params.index.values.tolist()

    for sub in sub_list:

        try:

            gsr_pd = pd.read_csv(resample_path + sub + '/parameters_gsr.csv')
            no_gsr_pd = pd.read_csv(resample_path + sub + '/parameters_no_gsr.csv')
            g_corr_x.append(float(gsr_pd['corr_x'][0]))
            g_corr_y.append(float(gsr_pd['corr_y'][0]))
            g_rmse_x.append(float(gsr_pd['rmse_x'][0]))
            g_rmse_y.append(float(gsr_pd['rmse_y'][0]))
            n_corr_x.append(float(no_gsr_pd['corr_x'][0]))
            n_corr_y.append(float(no_gsr_pd['corr_y'][0]))
            n_rmse_x.append(float(no_gsr_pd['rmse_x'][0]))
            n_rmse_y.append(float(no_gsr_pd['rmse_y'][0]))

        except:

            continue

    g_index = ['GSR' for x in range(len(g_corr_x))]
    n_index = ['No GSR' for x in range(len(g_corr_x))]

    tot = np.concatenate([g_index, n_index])
    corr_x = np.concatenate([g_corr_x, n_corr_x])
    corr_y = np.concatenate([g_corr_y, n_corr_y])
    rmse_x = np.concatenate([g_rmse_x, n_rmse_x])
    rmse_y = np.concatenate([g_rmse_y, n_rmse_y])

    swarm_df = pd.DataFrame({'corr_x': corr_x, 'corr_y': corr_y, 'rmse_x': rmse_x, 'rmse_y': rmse_y, 'index': tot})

    plt.clf()
    ax = sns.swarmplot(x='index', y='corr_x', data=swarm_df)
    ax.set(title='Correlation Distribution for GSR vs Non-GSR for x')
    plt.savefig('/home/json/Desktop/peer/hbm_figures/corr_x_comparison.png')
    plt.clf()
    ax = sns.swarmplot(x='index', y='corr_y', data=swarm_df)
    ax.set(title='Correlation Distribution for GSR vs Non-GSR for y')
    plt.savefig('/home/json/Desktop/peer/hbm_figures/corr_y_comparison.png')
    plt.clf()
    ax = sns.swarmplot(x='index', y='rmse_x', data=swarm_df)
    ax.set(title='RMSE Distribution for GSR vs Non-GSR for x')
    plt.savefig('/home/json/Desktop/peer/hbm_figures/rmse_x_comparison.png')
    plt.clf()
    ax = sns.swarmplot(x='index', y='rmse_y', data=swarm_df)
    ax.set(title='RMSE Distribution for GSR vs Non-GSR for y')
    plt.savefig('/home/json/Desktop/peer/hbm_figures/rmse_y_comparison.png')

create_swarms()

# #############################################################################
# Correlation matrix

resample_path = '/data2/Projects/Jake/Human_Brain_Mapping/'
params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
params = params.convert_objects(convert_numeric=True)
params = params[(params.scan_count == 3) | (params.scan_count == 2)]
sub_list = params.index.values.tolist()

corr_matrix_tp_x = []
corr_matrix_dm_x = []
corr_matrix_tp_y = []
corr_matrix_dm_y = []
count = 0

for sub in sub_list:

    try:

        if count == 0:

            expected_value = len(pd.read_csv(resample_path + sub + '/tppredictions.csv')['x_pred'])
            count += 1

        tp_x = np.array(pd.read_csv(resample_path + sub + '/tppredictions.csv')['x_pred'])
        tp_y = np.array(pd.read_csv(resample_path + sub + '/tppredictions.csv')['y_pred'])
        dm_x = np.array(pd.read_csv(resample_path + sub + '/dmpredictions.csv')['x_pred'][:250])
        dm_y = np.array(pd.read_csv(resample_path + sub + '/dmpredictions.csv')['y_pred'][:250])

        if (len(tp_x) == expected_value) & (len(dm_x) == expected_value):

            corr_matrix_tp_x.append(tp_x)
            corr_matrix_dm_x.append(tp_y)
            corr_matrix_tp_y.append(dm_x)
            corr_matrix_dm_y.append(dm_y)

    except:

        continue

x_matrix = np.concatenate([corr_matrix_tp_x, corr_matrix_dm_x])
y_matrix = np.concatenate([corr_matrix_tp_y, corr_matrix_dm_y])

from pylab import pcolor, show, colorbar, xticks, yticks

corr_matrix_x = np.corrcoef(x_matrix)
corr_matrix_y = np.corrcoef(y_matrix)
pcolor(corr_matrix_y)
colorbar()
show()

# #############################################################################
# Generalizable classifier


# #############################################################################
# Misc

def general_classifier(reg_list):

    funcTime = datetime.now()

    train_vectors1 = []
    train_vectors2 = []
    test_vectors = []

    for sub in reg_list[:train_set_count]:

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

            print('Initial average: ' + str(np.average(item)))
            item = mean_center_var_norm(item)
            print('Mean centered average: ' + str(np.average(item)))
            item = gs_regress(item, 0, item.shape[0]-1, 0, item.shape[1]-1, 0, item.shape[2]-1)
            print('GSR average: ' + str(np.average(item)))

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


# #############################################################################
# Turning each set of weights into a Nifti image for coefficient of variation map analysis

# reg_list = ['sub-5986705','sub-5375858','sub-5292617','sub-5397290','sub-5844932','sub-5787700','sub-5797959',
#             'sub-5378545','sub-5085726','sub-5984037','sub-5076391','sub-5263388','sub-5171285',
#             'sub-5917648','sub-5814325','sub-5169146','sub-5484500','sub-5481682','sub-5232535','sub-5905922',
#             'sub-5975698','sub-5986705','sub-5343770']
#
# train_set_count = len(reg_list) - 1
# resample_path = '/data2/Projects/Jake/Resampled/'
# eye_mask = nib.load('/data2/Projects/Jake/Resampled/eye_all_sub.nii.gz')
# eye_mask = eye_mask.get_data()
# for sub in reg_list:
#
#     train_vectors1 = []
#     train_vectors2 = []
#     test_vectors = []
#
#     print('starting participant ' + str(sub))
#
#     scan1 = nib.load(resample_path + sub + '/peer1_eyes.nii.gz')
#     scan1 = scan1.get_data()
#     scan2 = nib.load(resample_path + sub + '/peer2_eyes.nii.gz')
#     scan2 = scan2.get_data()
#     scan3 = nib.load(resample_path + sub + '/peer3_eyes.nii.gz')
#     scan3 = scan3.get_data()
#
#     for item in [scan1, scan2, scan3]:
#
#         for vol in range(item.shape[3]):
#             output = np.multiply(eye_mask, item[:, :, :, vol])
#
#             item[:, :, :, vol] = output
#
#     for item in [scan1, scan2, scan3]:
#         print('Initial average: ' + str(np.average(item)))
#         item = mean_center_var_norm(item)
#         print('Mean centered average: ' + str(np.average(item)))
#         item = gs_regress(item, 0, item.shape[0] - 1, 0, item.shape[1] - 1, 0, item.shape[2] - 1)
#         print('GSR average: ' + str(np.average(item)))
#
#     listed1 = []
#     listed2 = []
#     listed_testing = []
#
#     print('beginning vectors')
#
#     for tr in range(int(scan1.shape[3])):
#         tr_data1 = scan1[:, :, :, tr]
#         vectorized1 = np.array(tr_data1.ravel())
#         listed1.append(vectorized1)
#
#         tr_data2 = scan3[:, :, :, tr]
#         vectorized2 = np.array(tr_data2.ravel())
#         listed2.append(vectorized2)
#
#         te_data = scan2[:, :, :, tr]
#         vectorized_testing = np.array(te_data.ravel())
#         listed_testing.append(vectorized_testing)
#
#     train_vectors1.append(listed1)
#     test_vectors.append(listed_testing)
#     train_vectors2.append(listed2)
#
#     full_train1 = []
#     full_test = []
#     full_train2 = []
#
#     for part in range(len(reg_list[:train_set_count])):
#         for vect in range(scan1.shape[3]):
#             full_train1.append(train_vectors1[part][vect])
#             full_test.append(test_vectors[part][vect])
#             full_train2.append(train_vectors2[part][vect])
#
#         # train_vectors1 = np.asarray(listed1)
#         # test_vectors = np.asarray(listed_testing)
#         # train_vectors2 = np.asarray(listed2)
#
#         # #############################################################################
#         # Averaging training signal
#
#     print('average vectors')
#
#     train_vectors = data_processing(3, full_train1, full_train2)
#
#     # #############################################################################
#     # Import coordinates for fixations
#
#     print('importing fixations')
#
#     fixations = pd.read_csv('stim_vals.csv')
#     x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), len(reg_list[:train_set_count])) * monitor_width / 2, 3 - 1)
#     y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), len(reg_list[:train_set_count])) * monitor_height / 2,
#                         3 - 1)
#
#     # #############################################################################
#     # Create SVR Model
#
#     x_model, y_model = create_model(train_vectors, x_targets, y_targets)
#
#     x_model_coef = x_model.coef_
#     y_model_coef = y_model.coef_
#
#     x_model_coef = np.array(x_model_coef).reshape((35, 17, 14))
#     y_model_coef = np.array(y_model_coef).reshape((35, 17, 14))
#
#     img = nib.Nifti1Image(x_model_coef, np.eye(4))
#     img.header['pixdim'] = np.array([-1, 3, 3, 3, .80500031, 0, 0, 0])
#     img.to_filename('/data2/Projects/Jake/weights_coef/' + str(sub) + 'x.nii.gz')
#     img = nib.Nifti1Image(y_model_coef, np.eye(4))
#     img.header['pixdim'] = np.array([-1, 3, 3, 3, .80500031, 0, 0, 0])
#     img.to_filename('/data2/Projects/Jake/weights_coef/' + str(sub) + 'y.nii.gz')



# #############################################################################
# Creating a coefficient of variation map

# total = nib.load('/data2/Projects/Jake/weights_coef/totalx.nii.gz')
# data = total.get_data()
#
# coef_array = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
#
# for x in range(data.shape[0]):
#     for y in range(data.shape[1]):
#         for z in range(data.shape[2]):
#
#             vmean = np.mean(np.array(data[x, y, z, :]))
#             vstdev = np.std(np.array(data[x, y, z, :]))
#
#             for time in range(data.shape[3]):
#                 if np.round(vmean, 2) == 0.00:
#                     coef_array[x, y, z] = float(vstdev)
#                 else:
#                     coef_array[x, y, z] = float(vstdev)
#
# img = nib.Nifti1Image(coef_array, np.eye(4))
# img.header['pixdim'] = np.array([-1, 3, 3, 3, .80500031, 0, 0, 0])
# img.to_filename('/data2/Projects/Jake/weights_coef/x_coef_map_stdev.nii.gz')
#
# modified = nib.load('/data2/Projects/Jake/weights_coef/x_coef_map.nii.gz')
# data = modified.get_data()
#
# for x in range(data.shape[0]):
#     for y in range(data.shape[1]):
#         for z in range(data.shape[2]):
#             if abs(data[x, y, z]) > 100 or abs(data[x, y, z] < 3) and abs(np.round(data[x, y, z],2) != 0.00):
#                 data[x, y, z] = 1
#             else:
#                 data[x, y, z] = 0
#
# img = nib.Nifti1Image(data, np.eye(4))
# img.header['pixdim'] = np.array([-1, 3, 3, 3, .80500031, 0, 0, 0])
# img.to_filename('/data2/Projects/Jake/eye_masks/x_coef_map_eyes_100_5.nii.gz')

# #############################################################################
# Get distribution of voxel intensities from isolated eye coefficient of variation map to determine intensity threshold

# coef_sub = nib.load('/data2/Projects/Jake/weights_coef/x_coef_map.nii.gz')
# data = coef_sub.get_data()
#
# data_rav = data.ravel()
# data_rav = np.nan_to_num(data_rav)
# data_rav = np.array([x for x in data_rav if x != 0])
#
# xbins = np.histogram(data_rav, bins=300)[1]
#
# values, base = np.histogram(data_rav, bins=30)
# cumulative = np.cumsum(values)
#
# plt.figure()
# plt.hist(data_rav, xbins, color='b')
# plt.title('Full Raw')
# # plt.savefig('/home/json/Desktop/peer/eye_distr.png')
# plt.show()
# # plt.plot(base[:-1], cumulative/len(data_rav), color='g')
# # plt.show()

# #############################################################################
# Determine percentiles

# values, base = np.histogram(data_rav, bins=len(data_rav))
# cumulative = np.cumsum(values)/len(data_rav)
#
# for num in range(len(data_rav)):
#     if np.round(cumulative[num], 3) == .05:
#         print(base[num])
#
# # value_of_interest = base[percentile]

# #############################################################################
# Visualize error vs motion

# params = pd.read_csv('peer_didactics.csv', index_col='subject')
# params = params[params['x_gsr'] < 50000][params['y_gsr'] < 50000][params['mean_fd'] < 3.8][params['dvars'] < 1.5]
#
# # Need to fix script to not rely on indexing and instead include a subset based on mean and stdv parameters
# num_part = len(params)
#
# x_error_list = params.loc[:, 'x_gsr'][:num_part].tolist()
# y_error_list = params.loc[:, 'y_gsr'][:num_part].tolist()
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

