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
import pickle
import seaborn as sns
import matplotlib.ticker as ticker
from pylab import pcolor, show, colorbar
from sklearn.metrics import mean_squared_error, r2_score

from joblib import Parallel, delayed

monitor_width = 1680
monitor_height = 1050

# eye_mask = nib.load('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_eye_mask.nii.gz')
eye_mask = nib.load('/data2/Projects/Jake/eye_masks/2mm_eye_corrected.nii.gz')
eye_mask = eye_mask.get_data()
resample_path = '/data2/Projects/Jake/Human_Brain_Mapping/'

def load_data(min_scan=2):

    params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
    params = params.convert_objects(convert_numeric=True)

    if min_scan == 2:

        params = params[(params.scan_count == 2) | (params.scan_count == 3)]

    elif min_scan == 3:

        params = params[params.scan_count == 3]

    sub_list = params.index.values.tolist()

    return sub_list

def three_valid(sub, gsr_, second_file, viewtype):

    if viewtype == 'calibration+++++':

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

        x_model = pickle.load(open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/x_no_gsr_model.sav', 'rb'))
        y_model = pickle.load(open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/y_no_gsr_model.sav', 'rb'))

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

    if viewtype == 'calibration+++++++++++++++++++++++++++++++++':

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

        x_model = pickle.load(open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/x_no_gsr_train1_model.sav', 'rb'))
        y_model = pickle.load(open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/y_no_gsr_train1_model.sav', 'rb'))

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

    param_dict = {'sub': [sub, sub], 'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []}
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
            df_p.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/parameters_no_gsr_train1.csv')
            df_o = pd.DataFrame(output_dict)
            df_o.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/predictions_no_gsr_train1.csv')

            pickle.dump(x_model, open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/x_gsr_model_train1.sav', 'wb'))
            pickle.dump(y_model, open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/y_gsr_model_train1.sav', 'wb'))

        else:

            df_p = pd.DataFrame(param_dict)
            df_p.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/parameters_no_gsr_train1.csv')
            df_o = pd.DataFrame(output_dict)
            df_o.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/predictions_no_gsr_train1.csv')

            # pickle.dump(x_model, open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/x_no_gsr_train13_model.sav', 'wb'))
            # pickle.dump(y_model, open('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/y_no_gsr_train13_model.sav', 'wb'))

    else:

        output_dict['x_pred'] = predicted_x
        output_dict['y_pred'] = predicted_y

        df_o = pd.DataFrame(output_dict)
        df_o.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + str(sub) + '/' + str(save_name))

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

        scan_count = 2
        viewtype = 'calibration'
        gsr_ = 0

        if (viewtype == 'calibration') & (gsr_ == 1):
            second_file = '/peer2_eyes_sub.nii.gz'
            param_name = 'gsr_params.csv'
            save_name = 'gsr_pred.csv'
        elif (viewtype == 'calibration') & (gsr_ == 0):
            second_file = '/peer2_eyes_sub.nii.gz'
            param_name = 'no_gsr_train1_params.csv'
            save_name = 'no_gsr_train1_pred.csv'
        elif viewtype == 'tp':
            second_file = '/movie_TP_eyes_sub.nii.gz'
            save_name = 'no_gsr_train1_tp_pred.csv'
        elif viewtype == 'dm':
            second_file = '/movie_DM_eyes_sub.nii.gz'
            save_name = 'no_gsr_train1_dm_pred.csv'

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

sub_list = load_data(min_scan=3)

# Parallel(n_jobs=25)(delayed(peer_hbm)(sub, viewtype='calibration', gsr_='off')for sub in sub_list)
# Parallel(n_jobs=15)(delayed(peer_hbm)(sub, viewtype='tp', gsr_='off')for sub in sub_list)
# Parallel(n_jobs=15)(delayed(peer_hbm)(sub, viewtype='dm', gsr_='off')for sub in sub_list)


def pred_aggregate(gsr_status='off', viewtype='calibration', motion_type='mean_fd'):

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

    cov = pd.read_csv('Peer_pheno.csv', index_col='SubID')
    cov = cov.convert_objects(convert_numeric=True)
    cov = cov.sort_values(by='SWAN_HY', ascending=True)
    cov_t = cov.index.values.tolist()

    cov_t2 = ['sub-' + str(x) for x in cov_t]
    cov_t3 = [x for x in cov_t2 if x in sub_list]
    sub_list = cov_t3

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

# x_hm, y_hm = pred_aggregate(gsr_status='off', viewtype='tp', motion_type='mean_fd')


def save_heatmap(model, outname):

    sns.set()
    plt.clf()
    ax = sns.heatmap(model)
    ax.set(xlabel='Volumes', ylabel='Subjects')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
    # plt.savefig('/home/json/Desktop/peer/hbm_figures/' + outname + '.png', dpi=600)
    plt.show()


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
    sub_list = [x for x in sub_list if x not in partial_brain_list]

    for sub in sub_list:

        try:

            gsr_pd = pd.read_csv(resample_path + sub + '/parameters_gsr.csv')
            no_gsr_pd = pd.read_csv(resample_path + sub + '/parameters_no_gsr.csv')
            two_gsr_pd = pd.read_csv(resample_path + sub + '/parameters_gsr_two_scans.csv')
            g_corr_x.append(float(gsr_pd['corr_x'][0]))
            g_corr_y.append(float(gsr_pd['corr_y'][0]))
            g_rmse_x.append(float(gsr_pd['rmse_x'][0]))
            g_rmse_y.append(float(gsr_pd['rmse_y'][0]))
            n_corr_x.append(float(two_gsr_pd['corr_x'][0]))
            n_corr_y.append(float(two_gsr_pd['corr_y'][0]))
            n_rmse_x.append(float(two_gsr_pd['rmse_x'][0]))
            n_rmse_y.append(float(two_gsr_pd['rmse_y'][0]))

        except:

            continue

    g_index = ['Scan 1 & 3' for x in range(len(g_corr_x))]
    n_index = ['Scan 1' for x in range(len(g_corr_x))]

    tot = np.concatenate([g_index, n_index])
    corr_x = np.concatenate([g_corr_x, n_corr_x])
    corr_y = np.concatenate([g_corr_y, n_corr_y])
    rmse_x = np.concatenate([g_rmse_x, n_rmse_x])
    rmse_y = np.concatenate([g_rmse_y, n_rmse_y])

    swarm_df = pd.DataFrame({'corr_x': corr_x, 'corr_y': corr_y, 'rmse_x': rmse_x, 'rmse_y': rmse_y, 'index': tot})

    plt.clf()
    ax = sns.swarmplot(x='index', y='corr_x', data=swarm_df)
    ax.set(title='Correlation Distribution for Different Training Sets in x')
    plt.savefig('/home/json/Desktop/peer/hbm_figures/corr_x_comparison_scans.png')
    plt.clf()
    ax = sns.swarmplot(x='index', y='corr_y', data=swarm_df)
    ax.set(title='Correlation Distribution for Different Training Sets in y')
    plt.savefig('/home/json/Desktop/peer/hbm_figures/corr_y_comparison_scans.png')
    plt.clf()
    ax = sns.swarmplot(x='index', y='rmse_x', data=swarm_df)
    ax.set(title='RMSE Distribution for Different Training Sets in x')
    plt.savefig('/home/json/Desktop/peer/hbm_figures/rmse_x_comparison_scans.png')
    plt.clf()
    ax = sns.swarmplot(x='index', y='rmse_y', data=swarm_df)
    ax.set(title='RMSE Distribution for Different Training Sets in y')
    plt.savefig('/home/json/Desktop/peer/hbm_figures/rmse_y_comparison_scans.png')


def create_corr_matrix():

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

    corr_matrix_x = np.corrcoef(x_matrix)
    corr_matrix_y = np.corrcoef(y_matrix)

    return corr_matrix_x, corr_matrix_y, corr_matrix_tp_x, corr_matrix_tp_y, corr_matrix_dm_x, corr_matrix_dm_y

corr_matrix_x, corr_matrix_y, tp_x, tp_y, dm_x, dm_y = create_corr_matrix()
pcolor(corr_matrix_x)
colorbar()
show()


def separate_grouping(in_mat):

    fv = int(len(in_mat[0]))
    sv = fv / 2

    wi_ss = []
    wo_ss = []

    for numx in range(fv):
        for numy in range(fv):

            if (numx > sv) and (numy > sv) and (numx != numy):

                wi_ss.append(in_mat[numx][numy])

            elif (numx < sv) and (numy < sv) and (numx != numy):

                wi_ss.append(in_mat[numx][numy])

            else:

                if numx != numy:

                    wo_ss.append(in_mat[numx][numy])

    return wi_ss, wo_ss


wi_ss_x, wo_ss_x = separate_grouping(corr_matrix_x)
wi_ss_y, wo_ss_y = separate_grouping(corr_matrix_y)

wi_label = ['Within' for x in range(len(wi_ss_x))]
wo_label = ['Without' for x in range(len(wo_ss_x))]

df_dict = {'x': np.concatenate([wi_ss_x, wo_ss_x]), 'y': np.concatenate([wi_ss_y, wo_ss_y]),
           'within_without_group': np.concatenate([wi_label, wo_label]),
           'x_ax': ['Naturalistic Viewing' for x in range(len(wi_ss_x) + len(wo_ss_x))]}

df = pd.DataFrame.from_dict(df_dict)

sns.set()
plt.title('Within and Between Movie Correlation Discriminability in y')
sns.violinplot(x='x_ax', y='y', hue='within_without_group', data=df, split=True,
               inner='quart', palette={'Within': 'b', 'Without': 'y'})
plt.savefig('/home/json/Desktop/peer/hbm_figures/within_between_y.png', dpi=600)

def grouping_ss(within, without):

    wi_mean = np.mean(within)
    wo_mean = np.mean(without)
    wi_stdv = np.std(within)
    wo_stdv = np.std(without)

    return wi_mean, wo_mean, wi_stdv, wo_stdv

wi_mean_x, wo_mean_x, wi_stdv_x, wo_stdv_x = grouping_ss(wi_ss_x, wo_ss_x)
wi_mean_y, wo_mean_y, wi_stdv_y, wo_stdv_y = grouping_ss(wi_ss_y, wo_ss_y)


def extract_corr_values():

    params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
    params = params.convert_objects(convert_numeric=True)
    params = params[params.scan_count == 3]
    sub_list = params.index.values.tolist()
    sub_list = [x for x in sub_list if x not in partial_brain_list]
    resample_path = '/data2/Projects/Jake/Human_Brain_Mapping/'

    x_dict = {'no_gsr_train1': [], 'no_gsr_train3': [], 'no_gsr_train13': [], 'gsr_train1': []}
    y_dict = {'no_gsr_train1': [], 'no_gsr_train3': [], 'no_gsr_train13': [], 'gsr_train1': []}

    for sub in sub_list:

        try:

            x1 = pd.read_csv(resample_path + sub + '/parameters_no_gsr_train1.csv')['corr_x'][0]
            x2 = pd.read_csv(resample_path + sub + '/parameters_no_gsr_train3.csv')['corr_x'][0]
            x3 = pd.read_csv(resample_path + sub + '/parameters_no_gsr_train13.csv')['corr_x'][0]
            x4 = pd.read_csv(resample_path + sub + '/parameters_gsr_two_scans.csv')['corr_x'][0]

            x_dict['no_gsr_train1'].append(x1)
            x_dict['no_gsr_train3'].append(x2)
            x_dict['no_gsr_train13'].append(x3)
            x_dict['gsr_train1'].append(x4)

            y1 = pd.read_csv(resample_path + sub + '/parameters_no_gsr_train1.csv')['corr_y'][0]
            y2 = pd.read_csv(resample_path + sub + '/parameters_no_gsr_train3.csv')['corr_y'][0]
            y3 = pd.read_csv(resample_path + sub + '/parameters_no_gsr_train13.csv')['corr_y'][0]
            y4 = pd.read_csv(resample_path + sub + '/parameters_gsr_two_scans.csv')['corr_y'][0]

            y_dict['no_gsr_train1'].append(y1)
            y_dict['no_gsr_train3'].append(y2)
            y_dict['no_gsr_train13'].append(y3)
            y_dict['gsr_train1'].append(y4)

        except:

            print(sub + ' not successful')

    return x_dict, y_dict


def training_set_optimization():

    x_dict, y_dict = extract_corr_values()

    val_range = np.linspace(np.nanmin(x_dict['no_gsr_train1']), np.nanmax(x_dict['no_gsr_train1']), 1000)
    m1, b1 = np.polyfit(x_dict['no_gsr_train1'], x_dict['no_gsr_train13'], 1)

    r2_text = 'r2 value: ' + str(r2_score(x_dict['no_gsr_train1'], x_dict['no_gsr_train13']))[:5]

    plt.figure()
    plt.title('Comparison of Model Performance Based on Training Set in x')
    plt.xlabel('Scan 1 Correlation Values')
    plt.ylabel('Scan 1&3 Correlation Values')
    plt.scatter(x_dict['no_gsr_train1'], x_dict['no_gsr_train13'], label='Correlation values')
    plt.plot(val_range, m1*val_range + b1, color='r', label=r2_text)
    plt.plot([-.5, 1], [-.5, 1], '--', color='k', label='Identical Performance')
    plt.legend()
    plt.savefig('/home/json/Desktop/peer/hbm_figures/training_optimization_x1.png', dpi=600)
    plt.show()

    val_range = np.linspace(np.nanmin(y_dict['no_gsr_train1']), np.nanmax(y_dict['no_gsr_train1']), 1000)
    m2, b2 = np.polyfit(y_dict['no_gsr_train1'], y_dict['no_gsr_train13'], 1)

    r2_text = 'r2 value: ' + str(r2_score(y_dict['no_gsr_train1'], y_dict['no_gsr_train13']))[:5]

    plt.figure()
    plt.title('Comparison of Model Performance Based on Training Set in y')
    plt.xlabel('Scan 1 Correlation Values')
    plt.ylabel('Scan 1&3 Correlation Values')
    plt.scatter(y_dict['no_gsr_train1'], y_dict['no_gsr_train13'], label='Correlation values')
    plt.plot(val_range, m2*val_range + b2, color='r', label=r2_text)
    plt.plot([-.5, 1], [-.5, 1], '--', color='k', label='Identical Performance')
    plt.legend()
    plt.savefig('/home/json/Desktop/peer/hbm_figures/training_optimization_y1.png', dpi=600)
    plt.show()

    val_range = np.linspace(np.nanmin(x_dict['no_gsr_train3']), np.nanmax(x_dict['no_gsr_train3']), 1000)
    m1, b1 = np.polyfit(x_dict['no_gsr_train3'], x_dict['no_gsr_train13'], 1)

    r2_text = 'r2 value: ' + str(r2_score(x_dict['no_gsr_train3'], x_dict['no_gsr_train13']))[:5]

    plt.figure()
    plt.title('Comparison of Model Performance Based on Training Set in x')
    plt.xlabel('Scan 3 Correlation Values')
    plt.ylabel('Scan 1&3 Correlation Values')
    plt.scatter(x_dict['no_gsr_train3'], x_dict['no_gsr_train13'], label='Correlation values')
    plt.plot(val_range, m1*val_range + b1, color='r', label=r2_text)
    plt.plot([-.5, 1], [-.5, 1], '--', color='k', label='Identical Performance')
    plt.legend()
    plt.savefig('/home/json/Desktop/peer/hbm_figures/training_optimization_x3.png', dpi=600)
    plt.show()

    val_range = np.linspace(np.nanmin(y_dict['no_gsr_train1']), np.nanmax(y_dict['no_gsr_train1']), 1000)
    m2, b2 = np.polyfit(y_dict['no_gsr_train3'], y_dict['no_gsr_train13'], 1)

    r2_text = 'r2 value: ' + str(r2_score(y_dict['no_gsr_train3'], y_dict['no_gsr_train13']))[:5]

    plt.figure()
    plt.title('Comparison of Model Performance Based on Training Set in y')
    plt.xlabel('Scan 3 Correlation Values')
    plt.ylabel('Scan 1&3 Correlation Values')
    plt.scatter(y_dict['no_gsr_train3'], y_dict['no_gsr_train13'], label='Correlation values')
    plt.plot(val_range, m2*val_range + b2, color='r', label=r2_text)
    plt.plot([-.5, 1], [-.5, 1], '--', color='k', label='Identical Performance')
    plt.legend()
    plt.savefig('/home/json/Desktop/peer/hbm_figures/training_optimization_y3.png', dpi=600)
    plt.show()


def gsr_comparison():

    x_dict, y_dict = extract_corr_values()

    val_range = np.linspace(np.nanmin(x_dict['no_gsr_train1']), np.nanmax(x_dict['no_gsr_train1']), 1000)
    m1, b1 = np.polyfit(x_dict['no_gsr_train1'], x_dict['gsr_train1'], 1)

    r2_text = 'r2 value: ' + str(r2_score(x_dict['no_gsr_train1'], x_dict['gsr_train1']))[:5]

    plt.figure()
    plt.title('Comparison of Model Performance Based on GSR in x')
    plt.xlabel('No-GSR Correlation Values')
    plt.ylabel('GSR Correlation Values')
    plt.scatter(x_dict['no_gsr_train1'], x_dict['gsr_train1'], label='Correlation values')
    plt.plot(val_range, m1 * val_range + b1, color='r', label=r2_text)
    plt.plot([-.5, 1], [-.5, 1], '--', color='k', label='Identical Performance')
    plt.legend()
    plt.savefig('/home/json/Desktop/peer/hbm_figures/gsr_comparison_x.png', dpi=600)
    plt.show()

    val_range = np.linspace(np.nanmin(y_dict['no_gsr_train1']), np.nanmax(y_dict['no_gsr_train1']), 1000)
    m2, b2 = np.polyfit(y_dict['no_gsr_train1'], y_dict['gsr_train1'], 1)

    r2_text = 'r2 value: ' + str(r2_score(y_dict['no_gsr_train1'], y_dict['gsr_train1']))[:5]

    plt.figure()
    plt.title('Comparison of Model Performance Based on GSR in y')
    plt.xlabel('No-GSR Correlation Values')
    plt.ylabel('GSR Correlation Values')
    plt.scatter(y_dict['no_gsr_train1'], y_dict['gsr_train1'], label='Correlation values')
    plt.plot(val_range, m2 * val_range + b2, color='r', label=r2_text)
    plt.plot([-.5, 1], [-.5, 1], '--', color='k', label='Identical Performance')
    plt.legend()
    plt.savefig('/home/json/Desktop/peer/hbm_figures/gsr_comparison_y.png', dpi=600)
    plt.show()


def threshold_proportion(threshold=.50, type='gsr'):

    x_fract = len([x for x in x_dict[type] if x > threshold]) / len(x_dict[type])
    y_fract = len([x for x in y_dict[type] if x > threshold]) / len(y_dict[type])

    print(x_fract, y_fract)



def motion_linear_regression(motion='dvars', dimension='corr_x'):

    params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
    params = params.convert_objects(convert_numeric=True)
    params = params[(params.scan_count == 3) | (params.scan_count == 2)]
    params = params.sort_values(by=[motion])
    sub_list = params.index.values.tolist()
    resample_path = '/data2/Projects/Jake/Human_Brain_Mapping/'

    sub_dict = {'dvars': [], 'mean_fd': [], 'corr_x': [], 'corr_y': []}

    for sub in sub_list:

        try:

            if (np.isnan(pd.read_csv(resample_path + sub + '/parameters_no_gsr.csv')['corr_x'][0]) is not False) or (np.isnan(pd.read_csv(resample_path + sub + '/parameters_no_gsr.csv')['corr_y'][0]) is not False):

                sub_dict['dvars'].append(params.loc[sub, 'dvars'])
                sub_dict['mean_fd'].append(params.loc[sub, 'mean_fd'])
                sub_dict['corr_x'].append(pd.read_csv(resample_path + sub + '/parameters_no_gsr.csv')['corr_x'][0])
                sub_dict['corr_y'].append(pd.read_csv(resample_path + sub + '/parameters_no_gsr.csv')['corr_y'][0])

            else:

                continue

        except:

            continue

    for item in [motion]:

        outliers_removed = []

        for x in sub_dict[item]:
            if abs(x) < np.nanmean(sub_dict[item]) + 3 * np.nanstd(sub_dict[item]):
                outliers_removed.append(x)
            else:
                outliers_removed.append('replaced')

    for item in ['corr_x', 'corr_y']:

        outliers_removed = []

        for x in range(len(sub_dict[motion])):

            if sub_dict[item][x] != 'replaced':

                outliers_removed.append(sub_dict[item][x])

            else:
                outliers_removed.append('replaced')

        sub_dict[item] = outliers_removed

    for item in [motion, 'corr_x', 'corr_y']:
        sub_dict[item] = [x for x in sub_dict[item] if x != 'replaced']

    z = np.polyfit(sub_dict[motion], sub_dict['corr_x'], 1)
    p = np.poly1d(z)

    input_range = np.linspace(np.nanmin(sub_dict[motion]), np.nanmax(sub_dict[motion]), 100)

    from scipy.stats import linregress

    print(linregress(sub_dict[motion], sub_dict[dimension]))

    plt.figure()
    plt.plot(np.array(sub_dict[motion]), np.array(sub_dict[dimension]), '.')
    plt.plot(input_range, p(input_range), '--')
    plt.show()


# for mm in ['mean_fd', 'dvars']:
#     for dim in ['corr_x', 'corr_y']:
#         print(mm, dim)
#         motion_linear_regression(motion=mm, dimension=dim)
#
#
#
#
# x_dict, y_dict = extract_corr_values()
#
# x_gsr = ttest_rel(x_dict['gsr'], x_dict['no-gsr'])
# x_scan = ttest_rel(x_dict['gsr'], x_dict['two_scan'])
# y_gsr = ttest_rel(y_dict['gsr'], y_dict['no-gsr'])
# y_scan = ttest_rel(y_dict['gsr'], y_dict['two_scan'])
#
# plt.hist(x_dict['gsr'], bins=np.arange(0, 1, .05))
# plt.show()
#
# from scipy.stats import wilcoxon
#
# x_gsr = wilcoxon(x_dict['gsr'], x_dict['no-gsr'])
# x_scan = wilcoxon(x_dict['gsr'], x_dict['two_scan'])
# y_gsr = wilcoxon(y_dict['gsr'], y_dict['no-gsr'])
# y_scan = wilcoxon(y_dict['gsr'], y_dict['two_scan'])
#
# import pyvttbl as pt
#
# x_values = np.concatenate([np.array(x_dict['gsr']), np.array(x_dict['no-gsr']), np.array(x_dict['two_scan'])]).tolist()
# y_values = np.concatenate([np.array(y_dict['gsr']), np.array(y_dict['no-gsr']), np.array(y_dict['two_scan'])]).tolist()
# gsr_status = np.concatenate([np.ones(len(x_dict['gsr'])), np.ones(len(x_dict['gsr'])), np.zeros(len(x_dict['gsr']))]).tolist()
# training_set = np.concatenate([np.ones(len(x_dict['gsr'])), np.zeros(len(x_dict['gsr'])), np.ones(len(x_dict['gsr']))]).tolist()
#
# anova_dict = {'x_corr': x_values, 'y_corr': y_values, 'gsr_status': gsr_status, 'training_set': training_set}
#
# df = pd.DataFrame.from_dict(anova_dict)
#
# aov = df.anova1way('x_corr', 'gsr_status')
# print(aov)

# #############################################################################
# SVM for binary classification

def bin_class():

    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix

    # Create all dm and tp vectors with targets
    # Divide into training and testing sets

    corr_matrix_x, corr_matrix_y, tp_x, tp_y, dm_x, dm_y = create_corr_matrix()

    dm_targets = ['dm' for x in range(len(dm_x))]
    tp_targets = ['tp' for x in range(len(tp_x))]

    tt_split = .50
    tt_split_list = np.linspace(.1, .9, num=9)
    split_val = int(np.round(tt_split * len(tp_x), 0))
    split_val_list = [int(np.round(tt_split * len(tp_x), 0)) for tt_split in tt_split_list]

    temp_analysis_x = []
    temp_analysis_y = []

    for split_val in [split_val]:

        train_data_x = np.concatenate([tp_x[:split_val], dm_x[:split_val]])
        test_data_x = np.concatenate([tp_x[split_val:], dm_x[split_val:]])
        train_data_y = np.concatenate([tp_y[:split_val], dm_y[:split_val]])
        test_data_y = np.concatenate([tp_y[split_val:], dm_y[split_val:]])
        train_targets = np.concatenate([tp_targets[:split_val], dm_targets[:split_val]])
        test_targets = np.concatenate([tp_targets[split_val:], dm_targets[split_val:]])

        clfx = svm.SVC(C=100, tol=.0001, kernel='linear', verbose=0)
        clfy = svm.SVC(C=100, tol=.0001, kernel='linear', verbose=0)

        clfx.fit(train_data_x, train_targets)
        predictions_x = clfx.predict(test_data_x)
        clfy.fit(train_data_y, train_targets)
        predictions_y = clfy.predict(test_data_y)

        temp_analysis_x.append(accuracy_score(predictions_x, test_targets))
        temp_analysis_y.append(accuracy_score(predictions_y, test_targets))

        print(confusion_matrix(predictions_x, test_targets))
        print(confusion_matrix(predictions_y, test_targets))


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

# fixations = pd.read_csv('stim_vals.csv')
# x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 1) * monitor_width / 2, 1)
# y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 1) * monitor_height / 2, 1)
#
# plt.figure()
# plt.scatter(x_targets, y_targets)
# plt.title('Calibration Screen')
# plt.savefig('/home/json/Desktop/peer/hbm_figures/calibration_screen.png', dpi=600)
# plt.show()

from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
params = params.convert_objects(convert_numeric=True)
sub_list = params.index.values.tolist()

data_path = '/data2/Projects/Lei/Peers/Prediction_data/'

dm_dist = []
tp_dist = []

prop_dm = []
prop_tp = []

for sub in sub_list:

    try:

        with open(data_path + sub + '/DM_eyemove.txt') as f:
            content = f.readlines()
        dm = [float(x.strip('\n')) for x in content if float(x.strip('\n')) < 2000]
        with open(data_path + sub + '/TP_eyemove.txt') as f:
            content = f.readlines()
        tp = [float(x.strip('\n')) for x in content if float(x.strip('\n')) < 2000]

        n_bins = 10
        threshold = 300

        prop_dm.append(len([x for x in dm if x < threshold]) / len(dm))
        prop_tp.append(len([x for x in tp if x < threshold]) / len(tp))

        mu, sigma = norm.fit(dm)
        n, bins_dm = np.histogram(dm, n_bins, normed=True)
        y_dm = mlab.normpdf(np.linspace(0, 300, 100), mu, sigma)

        mu, sigma = norm.fit(tp)
        n, bins_tp = np.histogram(tp, n_bins, normed=True)
        y_tp = mlab.normpdf(np.linspace(0, 300, 100), mu, sigma)

        if y_dm[0] < .01:

            dm_dist.append([np.linspace(0, 300, 100), y_dm])

        if y_tp[0] < .01:

            tp_dist.append([np.linspace(0, 300, 100), y_tp])

    except:

        continue

print(np.mean(prop_dm), np.mean(prop_tp))

figure_title = 'DM Eye Movement Magnitude Distributions'
plt.figure()
plt.title(figure_title)

for bins_, y_ in dm_dist:
    plt.plot(np.linspace(0, 300, 100), y_, linewidth=1, alpha=.5)
plt.xlim([0, 300])
# plt.savefig('/home/json/Desktop/' + figure_title + '.png', dpi=600)
plt.xlabel('Distance between 2-D fixation predictions')
plt.show()

above = []
mid = []
stdv = []
below = []

mean_ci_dm = [y for x,y in dm_dist]
mean_ci_tp = [y for x,y in tp_dist]

dm_mean = np.mean(mean_ci_dm, axis=0)
dm_below = dm_mean - 1.96 * np.std(mean_ci_dm, axis=0)
dm_above = dm_mean + 1.96 * np.std(mean_ci_dm, axis=0)

tp_mean = np.mean(mean_ci_dm, axis=0)
tp_below = tp_mean - 1.96 * np.std(mean_ci_tp, axis=0)
tp_above = tp_mean + 1.96 * np.std(mean_ci_tp, axis=0)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals

    figure_title = 'TP Eye Movement Magnitude Distributions'

    plt.fill_between(np.linspace(0, 300, 100), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    # plt.plot(mean, color_mean)
    plt.plot(np.linspace(0, 300, 100), mean)
    plt.title(figure_title)
    plt.xlabel('Distance between 2-D fixation predictions')
    plt.savefig('/home/json/Desktop/peer/Figures/' + figure_title + '.png', dpi=600)


plot_mean_and_CI(tp_mean, tp_below, tp_above, color_mean='b', color_shading='b')


###### Squashing function

def squash(in_val, c1, c2):

    output = c1 + (c2 - c1) * np.tanh((in_val - c1)/(c2 - c1))

    return output

params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
params = params.convert_objects(convert_numeric=True)
sub_list = params.index.values.tolist()

for sub in sub_list:

    try:

        # with open(data_path + sub + '/DM_eyemove.txt') as f:
        #     content = f.readlines()
        # dm = [float(x.strip('\n')) for x in content if float(x.strip('\n')) ]
        with open(data_path + sub + '/TP_eyemove.txt') as f:
            content = f.readlines()
        tp = [float(x.strip('\n')) for x in content if float(x.strip('\n'))]

        # dm_mean = np.mean(dm)
        tp_mean = np.mean(tp)
        # dm_std = np.std(dm)
        tp_std = np.std(tp)

        # dm_z = [(x-dm_mean)/dm_std for x in dm]
        tp_z = [(x-tp_mean)/tp_std for x in tp]

        t_upper = 4.0
        t_lower = 2.5

        dm_squashed = []
        tp_squashed = []
        dm_spikes = []
        tp_spikes = []

        # for x in dm_z:
        #     if abs(x) < t_lower:
        #         dm_squashed.append(x)
        #         dm_spikes.append(0)
        #     else:
        #         dm_squashed.append(squash(x, t_lower, t_upper))
        #         dm_spikes.append(1)

        for x in tp_z:
            if abs(x) < t_lower:
                tp_squashed.append(x)
                tp_spikes.append(0)
            else:
                tp_squashed.append(squash(x, t_lower, t_upper))
                tp_spikes.append(1)

        # dm_output = [dm_mean + x*dm_std for x in dm_squashed]
        tp_output = [tp_mean + x*tp_std for x in tp_squashed]

        # with open('/data2/Projects/Lei/Peers/Prediction_data/' + sub + '/DM_eyemove_squashed.txt', 'w') as dm_file:
        #     for item in dm_output:
        #         dm_file.write("%s\n" % item)

        with open('/data2/Projects/Lei/Peers/Prediction_data/' + sub + '/TP_eyemove_squashed.txt', 'w') as tp_file:
            for item in tp_output:
                tp_file.write("%s\n" % item)

        # with open('/data2/Projects/Lei/Peers/Prediction_data/' + sub + '/DM_eyemove_spikes.txt', 'w') as dm_file:
        #     for item in dm_spikes:
        #         dm_file.write("%s\n" % item)

        with open('/data2/Projects/Lei/Peers/Prediction_data/' + sub + '/TP_eyemove_spikes.txt', 'w') as tp_file:
            for item in tp_spikes:
                tp_file.write("%s\n" % item)

    except:

        continue




#################################### Eye Tracking Analysis

eye_tracking_path = '/data2/HBN/EEG/data_RandID/'
hbm_path = '/data2/Projects/Jake/Human_Brain_Mapping/'


def scale_x_pos(fixation):

    return (fixation - 400) * (1680 / 800)


def scale_y_pos(fixation):

    return (fixation - 300) * (1050 / 600)


def et_samples_to_pandas(sub):

    sub = sub.strip('sub-')

    with open(eye_tracking_path + sub + '/Eyetracking/txt/' + sub + '_Video4_Samples.txt') as f:
        reader = csv.reader(f, delimiter='\t')
        content = list(reader)[38:]

        headers = content[0]

        df = pd.DataFrame(content[1:], columns=headers, dtype='float')

    msg_time = list(df[df.Type == 'MSG'].Time)

    start_time = float(msg_time[2])
    end_time = float(msg_time[3])

    df_msg_removed = df[(df.Time > start_time) & (df.Time < end_time)][['Time',
                                                                        'R POR X [px]',
                                                                        'R POR Y [px]']]

    df_msg_removed.update(df_msg_removed['R POR X [px]'].apply(scale_x_pos))
    df_msg_removed.update(df_msg_removed['R POR Y [px]'].apply(scale_y_pos))

    return df_msg_removed, start_time, end_time


def average_fixations_per_tr(df, start_time, end_time):

    mean_fixations = []

    movie_volume_count = 250
    movie_TR = 800  # in milliseconds

    for num in range(movie_volume_count):

        bin0 = start_time + num * 1000 * movie_TR
        bin1 = start_time + (num + 1) * 1000 * movie_TR
        df_temp = df[(df.Time >= bin0) & (df.Time <= bin1)]

        x_pos = np.mean(df_temp['R POR X [px]'])
        y_pos = np.mean(df_temp['R POR Y [px]'])

        mean_fixations.append([x_pos, y_pos])

    df = pd.DataFrame(mean_fixations, columns=(['x_pred', 'y_pred']))

    return df


def save_mean_fixations(mean_df):

    mean_df.to_csv(hbm_path + sub + '/et_device_pred.csv')
    # df.to_csv('/home/json/Desktop/test.csv')



params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
params = params.convert_objects(convert_numeric=True)
sub_list = params.index.values.tolist()

sub_list_with_et_and_peer = []

for sub in sub_list:

    try:

        df_output, start_time, end_time = et_samples_to_pandas(sub)
        mean_df = average_fixations_per_tr(df_output, start_time, end_time)
        # save_mean_fixations(mean_df)

        print('Completed processing ' + sub)
        sub_list_with_et_and_peer.append(sub)

    except:

        print('Error processing ' + sub)


def compare_et_and_peer(sub, plot=False):

    df_output, start_time, end_time = et_samples_to_pandas(sub)
    mean_df = average_fixations_per_tr(df_output, start_time, end_time)

    peer_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/no_gsr_train1_tp_pred.csv')

    corr_val = compute_icc(mean_df['x_pred'], peer_df['x_pred'])

    if plot:

        plt.figure(figsize=(10,5))
        plt.plot(np.linspace(0, 249, 250), mean_df['x_pred'], 'r-', label='eye-tracker')
        plt.plot(np.linspace(0, 249, 250), peer_df['x_pred'], 'b-', label='PEER')
        plt.title(sub)
        plt.xlabel('TR')
        plt.ylabel('Fixation location (px)')
        plt.legend()
        plt.savefig('/home/json/Desktop/peer/et_peer_comparison/' + sub + '.png', dpi=1200)
        plt.show()

    return corr_val

# sample_list_with_both_et_and_peer = ['sub-5743805', 'sub-5783223']
# sub_list_with_et_and_peer
# subset_list = ['sub-5161062', 'sub-5127994', 'sub-5049983', 'sub-5036745', 'sub-5002891', 'sub-5041416']

corr_vals = []

for sub in sub_list_with_et_and_peer:
    try:
        temp_val = compare_et_and_peer(sub, plot=False)
        corr_vals.append(temp_val)
        print(sub + ' processing completed.')
    except:
        continue

index_vals = np.zeros(len(corr_vals))
swarm_df = pd.DataFrame({'corr_x': corr_vals, 'index':index_vals})

plt.clf()
ax = sns.swarmplot(x='index', y='corr_x', data=swarm_df)
ax.set(title='Correlation Distribution for Different Training Sets in x')
plt.show()