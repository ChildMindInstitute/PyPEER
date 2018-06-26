# -*- coding: utf-8 -*-
"""
Created on Sat May 26 11:02:35 2018

@author: json
"""

"""
Glossary

/home/json/Desktop/peer/model_outputs.csv - list of subjects with at least 2 good calibration scans


"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import matplotlib.ticker as ticker
from statsmodels.formula.api import ols

monitor_width = 1680
monitor_height = 1050

def fig_one():
    
    sns.set()    
    
    sub_list = pd.DataFrame.from_csv('/home/json/Desktop/peer/phenotypes_full.csv').index.tolist()
    
    x_dist = []
    y_dist = []
    
    for sub in sub_list:
        
        sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')
        corr_x = sub_df.corr_x[0]
        corr_y = sub_df.corr_y[0] 
        
        x_dist.append(corr_x)
        y_dist.append(corr_y)
    
    params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
    params = params[params.scan_count == '3']
    sub_list = params.index.tolist()
    
    x_dist_1 = []
    x_dist_3 = []
    x_dist_13 = []
    x_dist_1gsr = []
    
    for sub in sub_list:
        
        corr_x_1 = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')['corr_x'][0]
        corr_x_3 = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train3_model_parameters.csv')['corr_x'][0]
        corr_x_13 = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train13_model_parameters.csv')['corr_x'][0]
        corr_gsr = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr1_train1_model_parameters.csv')['corr_x'][0]
        
        x_dist_1.append(corr_x_1)
        x_dist_3.append(corr_x_3)
        x_dist_13.append(corr_x_13)
        x_dist_1gsr.append(corr_gsr)
    
    x_dist_1 = [x for x in x_dist_1 if ~np.isnan(x)]
    x_dist_3 = [x for x in x_dist_3 if ~np.isnan(x)]
    x_dist_13 = [x for x in x_dist_13 if ~np.isnan(x)]
    x_dist_1gsr = [x for x in x_dist_1gsr if ~np.isnan(x)]
    
    volume_censor_df = pd.read_csv('/home/json/Desktop/peer/all_censored_corr.csv')
    x_dist_vc = list(volume_censor_df.x_censor)
    
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    x_values, x_base = np.histogram(x_dist, bins=40)
    x_values = [0] + list(np.flip(x_values, 0)) + [0]
    x_base = [1] + list(np.flip(x_base, 0))
    x_cumulative = np.cumsum(x_values)
    
    plt.plot(x_base[:-1] + [-.6], x_cumulative/np.max(x_cumulative), c='blue', label='x-direction')
    y_values, y_base = np.histogram(y_dist, bins=40)
    y_values = [0] + list(np.flip(y_values, 0)) + [0]
    y_base = [1] + list(np.flip(y_base, 0))
    y_cumulative = np.cumsum(y_values)
    
    plt.plot(y_base[:-1] + [-.6], y_cumulative/np.max(y_cumulative), c='green', label='y-direction')
    plt.title("PEER1 Accuracy")
    plt.xlabel("Pearson's r")
    plt.ylabel("Cumulative Density")
    plt.legend(loc=2)
    plt.ylim([-.05, 1.05])
    plt.xlim([1.05, -.6])
    
    plt.subplot(122)
        
    x_1_values, x_1_base = np.histogram(x_dist_1, bins=40)
    x_1_values = [0] + list(np.flip(x_1_values, 0)) + [0]
    x_1_base = [1] + list(np.flip(x_1_base, 0))
    
    x_1_cumulative = np.cumsum(x_1_values)
    plt.plot(x_1_base[:-1] + [-.6], x_1_cumulative/np.nanmax(x_1_cumulative), c='blue', label='PEER1')
    
    x_3_values, x_3_base = np.histogram(x_dist_3, bins=40)
    x_3_values = [0] + list(np.flip(x_3_values, 0)) + [0]
    x_3_base = [1] + list(np.flip(x_3_base, 0))
    
    x_3_cumulative = np.cumsum(x_3_values)
    plt.plot(x_3_base[:-1] + [-.6], x_3_cumulative/np.nanmax(x_3_cumulative), c='red', label='PEER3_')
    
    x_13_values, x_13_base = np.histogram(x_dist_13, bins=40)
    x_13_values = [0] + list(np.flip(x_13_values, 0)) + [0]
    x_13_base = [1] + list(np.flip(x_13_base, 0))
    x_13_cumulative = np.cumsum(x_13_values)
    
    plt.plot(x_13_base[:-1] + [-.6], x_13_cumulative/np.nanmax(x_13_cumulative), c='purple', label='PEER1&3')
    
    plt.title("Comparison of Model Accuracy by Training Set")
    plt.xlabel("Pearson's r")
    plt.ylabel("Cumulative Density")
    plt.legend(loc=2)
    plt.ylim([-.05, 1.05])
    plt.xlim([1.05, -.6])
    plt.savefig('/home/json/Desktop/manuscript_figures/fig1.png', dpi=600)
    plt.show()
    
#    plt.subplot(133)
#    
#    x_1_values, x_1_base = np.histogram(x_dist_1, bins=40)
#    x_1_values = [0] + list(np.flip(x_1_values, 0)) + [0]
#    x_1_base = [1] + list(np.flip(x_1_base, 0))
#    
#    x_1_cumulative = np.cumsum(x_1_values)
#    plt.plot(x_1_base[:-1] + [-.6], x_1_cumulative/np.nanmax(x_1_cumulative), c='blue', label='PEER1_r')
#    
#    x_values_vc, x_base_vc = np.histogram(x_dist_vc, bins=40)
#
#    x_values_vc = [0] + list(np.flip(x_values_vc, 0)) + [0]
#    x_base_vc = [1] + list(np.flip(x_base_vc, 0))
#    x_cumulative_vc = np.cumsum(x_values_vc)
#    
#    plt.plot(x_base_vc[:-1] + [-.6], x_cumulative_vc/np.max(x_cumulative_vc), c='magenta', label='PEER1VC_r')    
#    
#    x_1gsr_values, x_1gsr_base = np.histogram(x_dist_1gsr, bins=40)
#    x_1gsr_values = [0] + list(np.flip(x_1gsr_values, 0)) + [0]
#    x_1gsr_base = [1] + list(np.flip(x_1gsr_base, 0))
#    x_1gsr_cumulative = np.cumsum(x_1gsr_values)
#    
#    plt.plot(x_1gsr_base[:-1] + [-.6], x_1gsr_cumulative/np.nanmax(x_1gsr_cumulative), '-', c='black', label='PEER1GSR_r')
#        
#    plt.title("Effects of Pre-Processing on PEER1 Accuracy")
#    plt.xlabel("Pearson's r")
#    plt.ylabel("Cumulative Density")
#    plt.legend(loc=2)
#    plt.ylim([-.05, 1.05])
#    plt.xlim([1.05, -.6])
    
    #plt.savefig('/home/json/Desktop/manuscript_figures/fig1.png', dpi=600)
    
    


def training_optimization_paired_ttest():  
    
    sub_list = pd.DataFrame.from_csv('/home/json/Desktop/peer/phenotypes_full.csv').index.tolist()

    x_dist_1 = []
    x_dist_3 = []
    x_dist_13 = []
    x_dist_1gsr = []
    
    y_dist_1 = []
    y_dist_3 = []
    y_dist_13 = []
    y_dist_1gsr = []
    
    for sub in sub_list:
        
        x_dist_1.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')['corr_x'][0])
        x_dist_1gsr.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr1_train1_model_parameters.csv')['corr_x'][0])
        y_dist_1.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')['corr_y'][0])
        y_dist_1gsr.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr1_train1_model_parameters.csv')['corr_y'][0])
        
        try:
            x_dist_13.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train13_model_parameters.csv')['corr_x'][0])
            y_dist_13.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train13_model_parameters.csv')['corr_y'][0])
        except:
            x_dist_13.append(np.nan)
            y_dist_13.append(np.nan)
        try:
            x_dist_3.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train3_model_parameters.csv')['corr_x'][0])
            y_dist_3.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train3_model_parameters.csv')['corr_y'][0])
        except:
            x_dist_3.append(np.nan)
            y_dist_3.append(np.nan)
        
    df_dict = {'x_1': x_dist_1, 'x_3': x_dist_3, 'x_13': x_dist_13, 'x_1gsr': x_dist_1gsr,
               'y_1': y_dist_1, 'y_3': y_dist_3, 'y_13': y_dist_13, 'y_1gsr': y_dist_1gsr}
    df = pd.DataFrame.from_dict(df_dict)
    
    for group1, group2 in [['1', '13'], ['3', '13'], ['1', '3']]:
        
        direction = 'x_'
        
        group1 = direction + group1
        group2 = direction + group2
        
        df_subset = df[[group1 ,group2]].dropna(axis=0)
        
        t_stat, p_val = ttest_rel(df_subset[group1].tolist(), df_subset[group2].tolist())
        print(group1, group2, t_stat, p_val)



def paired_ttest_cal_scan():
    
    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/phenotypes_full.csv')
    
    df_fd =  df[['Scan1_meanfd', 'Scan3_meanfd']].dropna(axis=0)
    print(stats.ttest_rel(list(df_fd.Scan1_meanfd), list(df_fd.Scan3_meanfd)))
    df_dvars = df[['Scan1_dvars', 'Scan3_dvars']].dropna(axis=0)
    print(stats.ttest_rel(list(df_dvars.Scan1_dvars), list(df_dvars.Scan3_dvars)))


def fig_two():
    
    sns.set()
    
    ###########
    # Mean FD

    params = pd.read_csv('/home/json/Desktop/peer/phenotypes_full.csv', index_col='Subject')

    viewtype = 'calibration'
    modality = 'peer'
    cscheme = 'inferno'
    sorted_by = 'Scan1_meanfd'

    x_stack = []
    y_stack = []

    params = params.sort_values(by=[sorted_by], ascending=True)
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)
    
    arr = np.zeros((filename_dict[viewtype]['num_vol']))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(x_targets)
        y_stack.append(y_targets)
        
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)
        y_stack.append(arry)
        
    x_hm = []
    y_hm = []

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)
            
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)
        y_stack.append(arry)            
            
    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(x_targets)
        y_stack.append(y_targets)

    x_hm = np.stack(x_stack)

    x_spacing = len(x_hm[0])
    
    plt.figure()
    plt.subplot(221)
    plt.title('Predictions Ordered by Mean FD')
    ax = sns.heatmap(x_hm, cmap=cscheme, xticklabels=False, yticklabels=False, cbar=False)
    ax.set_ylabel('Subjects')
    ax.set_xlabel('Volumes')
    
    ############
    # Age
    
    params = pd.read_csv('/home/json/Desktop/peer/phenotypes_full.csv', index_col='Subject')

    viewtype = 'calibration'
    modality = 'peer'
    cscheme = 'inferno'
    sorted_by = 'age'

    x_stack = []
    y_stack = []

    params = params.sort_values(by=[sorted_by], ascending=False)
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)
    
    arr = np.zeros((filename_dict[viewtype]['num_vol']))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(x_targets)
        y_stack.append(y_targets)
        
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)
        y_stack.append(arry)
        
    x_hm = []
    y_hm = []

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)
            
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)
        y_stack.append(arry)            
            
    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(x_targets)
        y_stack.append(y_targets)

    x_hm = np.stack(x_stack)

    x_spacing = len(x_hm[0])
    
    plt.subplot(222)
    plt.title('Predictions Ordered by Age')
    ax = sns.heatmap(x_hm, cmap=cscheme, xticklabels=False, yticklabels=False, cbar=False)
    ax.set_ylabel('Subjects')
    ax.set_xlabel('Volumes')
    
    
    ##### Linear Regression
    
    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/phenotypes_full.csv')
    
    plt.subplot(223)
    plt.title('Impact of Mean FD')
    ax = sns.regplot(x='Scan1_meanfd', y='corr_x', data=df)
    ax.set_ylabel("Pearson's r")
    ax.set_xlabel('Mean FD')
    plt.ylim([-.5, 1.05])
    
    plt.subplot(224)
    plt.title('Impact of Age')
    ax = sns.regplot(x='age', y='corr_x', data=df)
    ax.set_ylabel("Pearson's r")
    ax.set_xlabel('Age')
    plt.ylim([-.5, 1.05])    
    plt.tight_layout()
    
    plt.savefig('/home/json/Desktop/manuscript_figures/fig2.png', dpi=600)    
    
    plt.show()
    


def phenotype_motion():
    
    # Uses correlation values from PEER1

    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/phenotypes_full.csv')
    
    # In x-direction

    meanfd = ols(formula="corr_x ~ Scan1_meanfd", data=df).fit()
    print(meanfd.summary())
    dvars = ols(formula="corr_x ~ Scan1_dvars", data=df).fit()
    print(dvars.summary())
    age = ols(formula="corr_x ~ age", data=df).fit()
    print(age.summary())
    fsiq = ols(formula="corr_x ~ fsiq", data=df).fit()
    print(fsiq.summary())
    age_meanfd = ols(formula="corr_x ~ Scan1_meanfd + age", data=df).fit()
    print(age_meanfd.summary())
    
    # In y-direction
    
    meanfd = ols(formula="corr_y ~ Scan1_meanfd", data=df).fit()
    print(meanfd.summary())
    dvars = ols(formula="corr_y ~ Scan1_dvars", data=df).fit()
    print(dvars.summary())
    age = ols(formula="corr_y ~ age", data=df).fit()
    print(age.summary())
    fsiq = ols(formula="corr_y ~ fsiq", data=df).fit()
    print(fsiq.summary())
    age_meanfd = ols(formula="corr_y ~ Scan1_meanfd + age", data=df).fit()
    print(age_meanfd.summary())
    
    # Visual inspection for potential outliers (e.g. low motion, low model accuracy)

    plt.figure()
    sns.regplot(x='Scan1_meanfd', y='corr_x', data=df)
    plt.show()
    
    # Identification of outliers via thresholding for model accuracy and mean_fd
    
    outlier_df = df[(df.corr_x < 0) & (df.Scan1_meanfd < .2)][['corr_x', 'Scan1_meanfd', 'Scan2_meanfd']]



def fig_three():
    
    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/phenotypes_full.csv')
    sub_list = df.index.tolist()

    x_norm = []    
    x_gs = []
    
    y_norm = []
    y_gs = []
    
    for sub in sub_list:
        
        x_norm.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')['corr_x'][0])
        y_norm.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')['corr_y'][0])

        x_gs.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr1_train1_model_parameters.csv')['corr_x'][0])
        y_gs.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr1_train1_model_parameters.csv')['corr_y'][0])

    
    df_dict = {'x_norm': x_norm, 'x_gs': x_gs, 'y_norm': y_norm, 'y_gs': y_gs}    
    df = pd.DataFrame.from_dict(df_dict)
    
    x_t, x_p = ttest_rel(df.x_norm, df.x_gs)
    y_t, y_p = ttest_rel(df.y_norm, df.y_gs)
    
    print(x_t, x_p)
    print(y_t, y_p)
    
    vc_df = pd.read_csv('/home/json/Desktop/peer/all_censored_corr.csv')
    
    x_t, x_p = ttest_rel(vc_df.x_raw, vc_df.x_censor)
    y_t, y_p = ttest_rel(vc_df.y_raw, vc_df.y_censor)
    
    print(x_t, x_p)
    print(y_t, y_p)

    plt.figure(figsize=(10, 11))
    plt.suptitle('Impact of Preprocessing on Model Accuracy', fontweight='bold')
    plt.subplot(221)
    plt.title('x-direction (All Subjects)')
    plt.boxplot([np.array(df.x_norm), np.array(df.x_gs), np.array(vc_df.x_censor)])
    plt.xticks([1, 2, 3], ['Original', 'GSR', 'VC'])
    plt.ylim([-.5, 1.05])
    plt.subplot(222)
    plt.title('y-direction (All Subjects)')
    plt.boxplot([np.array(df.y_norm), np.array(df.y_gs), np.array(vc_df.y_censor)])
    plt.xticks([1, 2, 3], ['Original', 'GSR', 'VC'])
    plt.ylim([-.5, 1.05])
    #plt.savefig('/home/json/Desktop/manuscript_figures/fig3.png', dpi=600)
    
    #########
    # Remove subjects with mean FD < .2

    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/phenotypes_full.csv')
    df = df[df.Scan1_meanfd > .2]
    sub_list = df.index.tolist()

    x_norm = []    
    x_gs = []
    
    y_norm = []
    y_gs = []
    
    for sub in sub_list:
        
        x_norm.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')['corr_x'][0])
        y_norm.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')['corr_y'][0])

        x_gs.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr1_train1_model_parameters.csv')['corr_x'][0])
        y_gs.append(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr1_train1_model_parameters.csv')['corr_y'][0])

    
    df_dict = {'x_norm': x_norm, 'x_gs': x_gs, 'y_norm': y_norm, 'y_gs': y_gs}    
    df = pd.DataFrame.from_dict(df_dict)
    
    x_t, x_p = ttest_rel(df.x_norm, df.x_gs)
    y_t, y_p = ttest_rel(df.y_norm, df.y_gs)
    
    print(x_t, x_p)
    print(y_t, y_p)

    plt.subplot(223)
    plt.title('x-direction (Subjects with Mean FD > .2)')
    plt.boxplot([np.array(df.x_norm), np.array(df.x_gs)])
    plt.xticks([1, 2], ['Original', 'GSR'])
    plt.ylim([-.5, 1.05])
    plt.subplot(224)
    plt.title('y-direction (Subjects with Mean FD > .2)')
    plt.boxplot([np.array(df.y_norm), np.array(df.y_gs)])
    plt.xticks([1, 2], ['Original', 'GSR'])
    plt.ylim([-.5, 1.05])
    plt.savefig('/home/json/Desktop/manuscript_figures/fig3.png', dpi=600)
    plt.show()


def fig_four():
    
    sns.set()
    
    ###########
    # TP Heatmap

    params = pd.read_csv('/home/json/Desktop/peer/phenotypes_full.csv', index_col='Subject')

    viewtype = 'tp'
    modality = 'peer'
    cscheme = 'inferno'
    sorted_by = 'Scan1_meanfd'

    x_stack = []
    y_stack = []

    params = params.sort_values(by=[sorted_by], ascending=True)
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)
    
    arr = np.zeros((filename_dict[viewtype]['num_vol']))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])
        
    x_hm = []

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)
            
    avg_series_x = np.mean(x_stack, axis=0)
            
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.insert(0, arrx)        
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.insert(0, avg_series_x)
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)  
    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(avg_series_x)

    x_hm = np.stack(x_stack)
    
    plt.figure()
    grid = plt.GridSpec(6, 4)
    plt.subplot(grid[:2, :])
    plt.title('The Present')
    ax = sns.heatmap(x_hm, cmap=cscheme, xticklabels=False, yticklabels=False, cbar=False)
    ax.set_xlabel('Volumes')
    ax.set_ylabel('Subjects')
    
    ###########
    # DM Heatmap

    params = pd.read_csv('/home/json/Desktop/peer/phenotypes_full.csv', index_col='Subject')

    viewtype = 'dm'
    modality = 'peer'
    cscheme = 'inferno'
    sorted_by = 'Scan1_meanfd'

    x_stack = []
    y_stack = []

    params = params.sort_values(by=[sorted_by], ascending=True)
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)
    
    arr = np.zeros((filename_dict[viewtype]['num_vol']))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])
        
    x_hm = []

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)
            
    avg_series_x = np.mean(x_stack, axis=0)
    
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.insert(0, arrx)        
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.insert(0, avg_series_x)
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)  
    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(avg_series_x)

    x_hm = np.stack(x_stack)    
    
    plt.subplot(grid[2:4, :])
    plt.title('Despicable Me')
    ax = sns.heatmap(x_hm, cmap=cscheme, xticklabels=False, yticklabels=False, cbar=False)
    ax.set_xlabel('Volumes')
    ax.set_ylabel('Subjects')
    
    ###########
    # Movie correlations
    
    sub_list = list(pd.read_csv('/home/json/Desktop/peer/qap_naturalistic_mri.csv').subject)
    
    expected_value = 250

    pd_dict_x = {}
    pd_dict_y = {}

    corr_matrix_tp_x = []
    corr_matrix_tp_y = []

    for sub in sub_list:

        try:

            tp_x = np.array(pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')['x_pred'])
            tp_y = np.array(pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')['y_pred'])

            if len(tp_x) == expected_value:
                corr_matrix_tp_x.append(tp_x)
                corr_matrix_tp_y.append(tp_y)
    
                pd_dict_x[str('tp' + sub)] = tp_x
                pd_dict_y[str('tp' + sub)] = tp_y

        except:

            continue

    corr_matrix_dm_x = []
    corr_matrix_dm_y = []

    for sub in sub_list:

        try:
    
            dm_x = np.array(pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_dm_predictions.csv')['x_pred'][:250])
            dm_y = np.array(pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_dm_predictions.csv')['y_pred'][:250])
    
            if len(dm_x) == expected_value:
                corr_matrix_dm_x.append(dm_x)
                corr_matrix_dm_y.append(dm_y)
    
                pd_dict_x[str('dm' + sub)] = dm_x
                pd_dict_y[str('dm' + sub)] = dm_y
    
        except:
    
            continue

    pd_dict_x['index'] = range(len(pd_dict_x[str('dm' + sub)]))
    pd_dict_y['index'] = range(len(pd_dict_y[str('dm' + sub)]))

    df_x = pd.DataFrame.from_dict(pd_dict_x)
    df_x = df_x.set_index('index')
    df_x = df_x.reindex_axis(sorted(df_x.columns), axis=1)
    df_y = pd.DataFrame.from_dict(pd_dict_y)
    df_y = df_y.set_index('index')
    df_y = df_y.reindex_axis(sorted(df_y.columns), axis=1)

    corr_x = df_x.corr(method='pearson')
    corr_y = df_y.corr(method='pearson')

    # OPTIONAL - INCLUDE ONLY BELOW DIAGNOAL
    mask_x = np.zeros_like(corr_x)
    mask_x[np.triu_indices_from(mask_x)] = True
    mask_y = np.zeros_like(corr_y)
    mask_y[np.triu_indices_from(mask_y)] = True

    within_x = []
    between_x = []

    for item1 in corr_x.columns:
        for item2 in corr_x.columns:
            if (item1[:2] == item2[:2]) and (item1 != item2):
                within_x.append(corr_x[item1][item2])
            elif (item1[:2] != item2[:2]) and (item1 != item2):
                between_x.append(corr_x[item1][item2])

    print('Completed x matrix')

    final_dict_x = {}

    final_dict_x["Pearson's r"] = within_x + between_x
    final_dict_x[''] = ['Within Movie']*len(within_x) + ['Between Movie']*len(between_x)

    final_df_x = pd.DataFrame.from_dict(final_dict_x)
    
    within_y = []
    between_y = []

    for item1 in corr_y.columns:
        for item2 in corr_y.columns:
            if (item1[:2] == item2[:2]) and (item1 != item2):
                within_y.append(corr_y[item1][item2])
            elif (item1[:2] != item2[:2]) and (item1 != item2):
                between_y.append(corr_y[item1][item2])

    print('Completed y matrix')

    final_dict_y = {}

    final_dict_y["Pearson's r"] = within_y + between_y
    final_dict_y[''] = ['Within Movie']*len(within_y) + ['Between Movie']*len(between_y)

    final_df_y = pd.DataFrame.from_dict(final_dict_y)
    
    plt.subplot(grid[4:, :2])
    ax = sns.violinplot(x='', y="Pearson's r", data=final_df_x, scale='count')
    plt.ylim([-.5, 1])
    plt.title('Movie Correlations in x-')
    
    plt.subplot(grid[4:, 2:])
    ax = sns.violinplot(x='', y="Pearson's r", data=final_df_y, scale='count')
    plt.ylim([-.5, 1])
    plt.title('Movie Correlations in y-')

    plt.tight_layout()
    
    plt.savefig('/home/json/Desktop/manuscript_figures/fig4.png', dpi=600)
    
    plt.show()


 
def fig_three_old():

    sub_list = list(pd.read_csv('/home/json/Desktop/peer/qap_naturalistic_mri.csv').subject)
    
    expected_value = 250

    pd_dict_x = {}
    pd_dict_y = {}

    corr_matrix_tp_x = []
    corr_matrix_tp_y = []

    for sub in sub_list:

        try:

            tp_x = np.array(pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')['x_pred'])
            tp_y = np.array(pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')['y_pred'])

            if len(tp_x) == expected_value:
                corr_matrix_tp_x.append(tp_x)
                corr_matrix_tp_y.append(tp_y)
    
                pd_dict_x[str('tp' + sub)] = tp_x
                pd_dict_y[str('tp' + sub)] = tp_y

        except:

            continue

    corr_matrix_dm_x = []
    corr_matrix_dm_y = []

    for sub in sub_list:

        try:
    
            dm_x = np.array(pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_dm_predictions.csv')['x_pred'][:250])
            dm_y = np.array(pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_dm_predictions.csv')['y_pred'][:250])
    
            if len(dm_x) == expected_value:
                corr_matrix_dm_x.append(dm_x)
                corr_matrix_dm_y.append(dm_y)
    
                pd_dict_x[str('dm' + sub)] = dm_x
                pd_dict_y[str('dm' + sub)] = dm_y
    
        except:
    
            continue

    pd_dict_x['index'] = range(len(pd_dict_x[str('dm' + sub)]))
    pd_dict_y['index'] = range(len(pd_dict_y[str('dm' + sub)]))

    df_x = pd.DataFrame.from_dict(pd_dict_x)
    df_x = df_x.set_index('index')
    df_x = df_x.reindex_axis(sorted(df_x.columns), axis=1)
    df_y = pd.DataFrame.from_dict(pd_dict_y)
    df_y = df_y.set_index('index')
    df_y = df_y.reindex_axis(sorted(df_y.columns), axis=1)

    corr_x = df_x.corr(method='pearson')
    corr_y = df_y.corr(method='pearson')

    # OPTIONAL - INCLUDE ONLY BELOW DIAGNOAL
    mask_x = np.zeros_like(corr_x)
    mask_x[np.triu_indices_from(mask_x)] = True
    mask_y = np.zeros_like(corr_y)
    mask_y[np.triu_indices_from(mask_y)] = True

    within_x = []
    between_x = []

    for item1 in corr_x.columns:
        for item2 in corr_x.columns:
            if (item1[:2] == item2[:2]) and (item1 != item2):
                within_x.append(corr_x[item1][item2])
            elif (item1[:2] != item2[:2]) and (item1 != item2):
                between_x.append(corr_x[item1][item2])

    print('Completed x matrix')

    final_dict_x = {}

    final_dict_x["Pearson's r"] = within_x + between_x
    final_dict_x[''] = ['Within Movie']*len(within_x) + ['Between Movie']*len(between_x)

    final_df_x = pd.DataFrame.from_dict(final_dict_x)

    within_y = []
    between_y = []

    for item1 in corr_y.columns:
        for item2 in corr_y.columns:
            if (item1[:2] == item2[:2]) and (item1 != item2):
                within_y.append(corr_y[item1][item2])
            elif (item1[:2] != item2[:2]) and (item1 != item2):
                between_y.append(corr_y[item1][item2])

    print('Completed y matrix')

    final_dict_y = {}

    final_dict_y["Pearson's r"] = within_y + between_y
    final_dict_y[''] = ['Within Movie']*len(within_y) + ['Between Movie']*len(between_y)

    final_df_y = pd.DataFrame.from_dict(final_dict_y)

    plt.figure(figsize=(18, 15))
    plt.suptitle('Movie Discriminability', fontsize=16)
    plt.subplot(221)
    plt.title('Correlation Matrices for DM and TP')
    sns.heatmap(corr_x, cmap='inferno', xticklabels=False, yticklabels=False)
    plt.xticks([213, 607], ['DM', 'TP'])
    plt.yticks([213, 607], ['DM', 'TP'])

    plt.subplot(223)
    sns.heatmap(corr_y, cmap='inferno', xticklabels=False, yticklabels=False)
    plt.xticks([213, 607], ['DM', 'TP'])
    plt.yticks([213, 607], ['DM', 'TP'])

    plt.subplot(222)
    plt.title('Within and Between Movie Correlations')
    # ax = sns.boxplot(x='', y="Pearson's r", data=final_df_x)
    ax = sns.violinplot(x='', y="Pearson's r", data=final_df_x, scale='count')
    plt.ylim([-.5, 1])
    # ax = sns.stripplot(x='', y="Pearson's r", data=final_df_x, jitter=True)
    plt.subplot(224)
    ax = sns.violinplot(x='', y="Pearson's r", data=final_df_y, scale='count')
    plt.ylim([-.5, 1])
    plt.savefig('/home/json/Desktop/manuscript_figures/fig3.png', dpi=600)
    plt.show()
    
    
    
    
    
    
def fig_four_old():

    viewtype = 'tp'
    cscheme = 'inferno'

    def load_data(min_scan=2):
        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list


    params, sub_list = load_data(min_scan=2)

    x_stack = []
    y_stack = []
    sorted_by = 'mean_fd'

    params = params.sort_values(by=[sorted_by])
    sub_list = params.index.values.tolist()

    def create_sub_list_with_et_and_peer(full_list):

        """Creates a list of subjects with both ET and PEER predictions

        :param full_list: List of subject IDs containing all subjects with at least 2/3 valid calibration scans
        :return: Subject list with both ET and PEER predictions
        """

        et_list = []

        for sub in full_list:

            if (os.path.exists('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')) and \
                    (os.path.exists(
                        '/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')):
                et_list.append(sub)

        return et_list

    et_list = create_sub_list_with_et_and_peer(sub_list)
    with open('/data2/Projects/Lei/Peers/scripts/data_check/TP_bad_sub.txt') as f:
        bad_subs_list_tp = f.readlines()
    with open('/data2/Projects/Lei/Peers/scripts/data_check/DM_bad_sub.txt') as f:
        bad_subs_list_dm = f.readlines()

    bad_subs_list_tp = [x.strip('\n') for x in bad_subs_list_tp]
    bad_subs_list_dm = [x.strip('\n') for x in bad_subs_list_dm]

    et_list = [x for x in et_list if x not in bad_subs_list_tp]
    et_list = [x for x in et_list if x not in bad_subs_list_dm]


    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)

    modality = 'peer'

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv(
                    '/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (
                    len(y_series) == filename_dict[viewtype]['num_vol']):
                x_series = [x if abs(x) < monitor_width / 2 + .1 * monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height / 2 + .1 * monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)

    arr = np.zeros(len(x_series))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    if viewtype == 'calibration':

        for num in range(int(np.round(len(et_list) * .02, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        for num in range(int(np.round(len(et_list) * .02, 0))):
            x_stack.append(x_targets)
            y_stack.append(y_targets)

    else:

        avg_series_x = np.mean(x_stack, axis=0)
        avg_series_y = np.mean(y_stack, axis=0)

        for num in range(int(np.round(len(et_list) * .05, 0))):
            x_stack.append(arrx)
            y_stack.append(arry)

        # for num in range(int(np.round(len(et_list) * .02, 0))):
        #     x_stack.append(avg_series_x)
        #     y_stack.append(avg_series_y)

    modality = 'et'

    retain_et_list = []

    count_include = 0
    count_exclude = 0

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (
                    len(y_series) == filename_dict[viewtype]['num_vol']):
                x_series = [x if abs(x) < monitor_width / 2 + .1 * monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height / 2 + .1 * monitor_height else 0 for x in y_series]

                x_count = x_series.count(-840)
                y_count = y_series.count(525.0)

                if (x_count < 13) and (y_count < 13):

                    count_include += 1

                    print(sub, count_include)
                    retain_et_list.append(sub)

                    x_stack.append(x_series)
                    y_stack.append(y_series)

                else:

                    count_exclude += 1

                    print('Subject ' + sub + ' excluded from analysis', count_exclude)

        except:

            continue

    x_hm = np.stack(x_stack)
    y_hm = np.stack(y_stack)

    x_spacing = len(x_hm[0])

    plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(6, 8, hspace=.4)
    plt.subplot(grid[:3, :])
    plt.title('The Present Fixation Series in x-')
    ax = sns.heatmap(x_hm, cmap=cscheme, yticklabels=False, xticklabels=False)
    plt.yticks([200, 505], ['PEER', 'Eye-Tracking'])

    plt.subplot(grid[3:, :])
    plt.title('The Present Fixation Series in y-')
    ax = sns.heatmap(y_hm, cmap=cscheme, yticklabels=False)
    ax.set(xlabel='Volumes')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing / 5, 0)))
    plt.yticks([200, 505], ['PEER', 'Eye-Tracking'])
    #plt.savefig('/home/json/Desktop/manuscript_figures/fig4.png', dpi=600)
    plt.show()






    
    
    
    
    
def sup_fig_X():

    # To compare linear relationship between Pearson's r and factors that may affect head motion

    def load_data(min_scan=2):

        """Returns list of subjects with at least the specified number of calibration scans

        :param min_scan: Minimum number of scans required to be included in subject list
        :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
        :return: List containing subjects with at least min_scan calibration scans
        """

        params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
        params = params.convert_objects(convert_numeric=True)

        if min_scan == 2:

            params = params[(params.scan_count == 2) | (params.scan_count == 3)]

        elif min_scan == 3:

            params = params[params.scan_count == 3]

        sub_list = params.index.values.tolist()

        return params, sub_list

    params, sub_list = load_data(min_scan=2)

    motion_dict = {'mean_fd': [], 'dvars': [], 'corr_x': [], 'corr_y': [], 'age': [], 'fsiq': []}
    temp_df = pd.DataFrame.from_csv('/home/json/Desktop/peer/Peer_pheno.csv')
    temp_df = temp_df.drop_duplicates()

    for sub in sub_list:

        try:

            mean_fd = params.loc[sub, 'mean_fd']
            dvars = params.loc[sub, 'dvars']
            corr_x = pd.DataFrame.from_csv(resample_path + sub + '//gsr0_train1_model_parameters.csv')['corr_x'][0]
            corr_y = pd.DataFrame.from_csv(resample_path + sub + '//gsr0_train1_model_parameters.csv')['corr_y'][0]

            age_val = temp_df.loc[int(sub.strip('sub-')), 'Age']
            fsiq_val = temp_df.loc[int(sub.strip('sub-')), 'FSIQ']

            motion_dict['mean_fd'].append(mean_fd)
            motion_dict['dvars'].append(dvars)
            motion_dict['corr_x'].append(corr_x)
            motion_dict['corr_y'].append(corr_y)
            motion_dict['age'].append(age_val)
            motion_dict['fsiq'].append(fsiq_val)

        except:

            continue

    mean_fd_list = []
    dvars_list = []
    corr_x_list = []
    corr_y_list = []
    age_list = []
    fsiq_list = []

    motion_type = 'mean_fd'
    thresh = 100000

    for num in range(len(motion_dict[motion_type])):

        if motion_dict[motion_type][num] < thresh:

            mean_fd_list.append(motion_dict['mean_fd'][num])
            dvars_list.append(motion_dict['dvars'][num])
            corr_x_list.append(motion_dict['corr_x'][num])
            corr_y_list.append(motion_dict['corr_y'][num])
            age_list.append(motion_dict['age'][num])
            fsiq_list.append(motion_dict['fsiq'][num])

    motion_dict['mean_fd'] = mean_fd_list
    motion_dict['dvars'] = dvars_list
    motion_dict['corr_x'] = corr_x_list
    motion_dict['corr_y'] = corr_y_list
    motion_dict['age'] = age_list
    motion_dict['fsiq'] = fsiq_list

    ######## MeanFD and DVARS

    motion_type = 'mean_fd'

    val_range = np.linspace(np.nanmin(motion_dict[motion_type]), np.nanmax(motion_dict[motion_type]),
                            len(motion_dict[motion_type]))

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('MeanFD Effects on Model Accuracy')
    plt.xlabel('MeanFD')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(223)
    plt.xlabel('MeanFD')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    ########

    motion_type = 'dvars'

    val_range = np.linspace(np.nanmin(motion_dict[motion_type]), np.nanmax(motion_dict[motion_type]),
                            len(motion_dict[motion_type]))

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(222)
    plt.title('DVARS Effects on Model Accuracy')
    plt.xlabel('DVARS')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(224)
    plt.xlabel('DVARS')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})
    plt.savefig('/home/json/Desktop/peer_figures_final/motion_effects.png', dpi=600)
    plt.show()

    ######## Age and FSIQ

    motion_type = 'age'

    val_range = np.linspace(np.nanmin(motion_dict[motion_type]), np.nanmax(motion_dict[motion_type]),
                            len(motion_dict[motion_type]))

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('Age Effects on Model Accuracy')
    plt.xlabel('Age')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    slope, intercept, r_val, p_val, std_error = stats.linregress(motion_dict[motion_type], motion_dict['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(223)
    plt.xlabel('Age')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    ########

    motion_type = 'fsiq'

    fsiq_list = []
    corr_x_list = []
    corr_y_list = []

    for item in range(len(motion_dict['fsiq'])):

        if ~np.isnan(motion_dict['fsiq'][item]):
            fsiq_list.append(motion_dict['fsiq'][item])
            corr_x_list.append(motion_dict['corr_x'][item])
            corr_y_list.append(motion_dict['corr_y'][item])

    val_range = np.linspace(np.nanmin(motion_dict[motion_type]), np.nanmax(motion_dict[motion_type]),
                            len(motion_dict[motion_type]))

    slope, intercept, r_val, p_val, std_error = stats.linregress(fsiq_list, corr_x_list)
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(222)
    plt.title('FSIQ Effects on Model Accuracy')
    plt.xlabel('FSIQ')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_x'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})

    slope, intercept, r_val, p_val, std_error = stats.linregress(fsiq_list, corr_y_list)
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:5]
    r_text = 'r value: ' + str(r_val)[:5]

    plt.subplot(224)
    plt.xlabel('FSIQ')
    plt.ylabel("Pearson's r")
    plt.ylim([-.6, 1.0])
    plt.scatter(motion_dict[motion_type], motion_dict['corr_y'], alpha=.75)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.legend(loc=4, prop={'weight': 'bold', 'size': 9})
    plt.savefig('/home/json/Desktop/peer_figures_final/age_fsiq_effects.png', dpi=600)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def sup_fig_one():
    
    sns.set()
    
    params = pd.read_csv('model_outputs.csv', index_col='subject', dtype=object)
    params = params[params.scan_count == '3']
    sub_list = params.index.tolist()
    
    fixations = pd.read_csv('/home/json/Desktop/peer/stim_vals.csv')
    x_targets = np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2
    y_targets = np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2
    
    def create_dict_with_rmse_and_corr_values(sub_list):

        """Creates dictionary that contains list of rmse and corr values for all training combinations

        :return: Dictionary that contains list of rmse and corr values for all training combinations
        """

        file_dict = {'1': '/gsr0_train1_model_parameters.csv',
                     '3': '/gsr0_train3_model_parameters.csv',
                     '13': '/gsr0_train13_model_parameters.csv',
                     '1gsr': '/gsr1_train1_model_parameters.csv'}

        params_dict = {'1': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '3': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '13': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []},
                       '1gsr': {'corr_x': [], 'corr_y': [], 'rmse_x': [], 'rmse_y': []}}

        for sub in sub_list:

            for train_set in file_dict.keys():

                if os.path.exists('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + file_dict['3']) and (
                        np.isnan(pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + file_dict['3'])['corr_x'][0]) != True):

                    try:

                        temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + file_dict[train_set])
                        x_corr = temp_df['corr_x'][0]
                        y_corr = temp_df['corr_y'][0]
                        x_rmse = temp_df['rmse_x'][0]
                        y_rmse = temp_df['rmse_y'][0]

                        params_dict[train_set]['corr_x'].append(x_corr)
                        params_dict[train_set]['corr_y'].append(y_corr)
                        params_dict[train_set]['rmse_x'].append(x_rmse)
                        params_dict[train_set]['rmse_y'].append(y_rmse)

                    except:

                        print('Error processing subject ' + sub + ' for ' + train_set)

                else:

                    continue

        return params_dict

    params_dict = create_dict_with_rmse_and_corr_values(sub_list)

    #############
    x_ax = '1'
    y_ax = '13'

    val_range = np.linspace(-.6, 1.0, 480)
    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.figure(figsize=(15, 10))
    plt.suptitle("Comparing PEER's Predictive Accuracy Based on Training Set", fontsize=16)
    plt.subplot(2, 3, 1)
    plt.title('PEER1_r vs PEER1&3_r')
    plt.xlabel("PEER1_r")
    plt.ylabel("PEER1&3_r")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'], alpha=.5)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 4)
    plt.xlabel("PEER1_r")
    plt.ylabel("PEER1&3_r")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'], alpha=.5)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    #############
    x_ax = '3'
    y_ax = '13'

    val_range = np.linspace(-.6, 1.0, 480)
    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 2)
    plt.title('PEER3_r vs PEER1&3_r')
    plt.xlabel("PEER3_r")
    plt.ylabel("PEER1&3_r")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'], alpha=.5)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 5)
    plt.xlabel("PEER3_r")
    plt.ylabel("PEER1&3_r")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'], alpha=.5)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    #############
    x_ax = '1'
    y_ax = '3'

    val_range = np.linspace(-.6, 1.0, 480)
    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 3)
    plt.title('PEER1_r vs PEER3_r')
    plt.xlabel("PEER1_r")
    plt.ylabel("PEER3_r")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_x'], params_dict[y_ax]['corr_x'], alpha=.5)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})

    slope, intercept, r_val, p_val, std_error = stats.linregress(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'])
    r2_val = r_val**2
    r2_text = 'r2 value: ' + str(r2_val)[:4]
    r_text = 'r value: ' + str(r_val)[:4]

    plt.subplot(2, 3, 6)
    plt.xlabel("PEER1_r")
    plt.ylabel("PEER3_r")
    plt.xlim([-.6, 1.0])
    plt.ylim([-.6, 1.0])
    plt.scatter(params_dict[x_ax]['corr_y'], params_dict[y_ax]['corr_y'], alpha=.5)
    plt.plot(val_range, slope * (val_range) + intercept, color='r', label=r_text)
    plt.plot([-.6, 1], [-.6, 1], '--', color='k', label='Identical')
    plt.legend(loc=2, prop={'size': 9, 'weight': 'bold'})
    plt.savefig('/home/json/Desktop/manuscript_figures/sup_fig1.png', dpi=600)
    plt.show()

    
    
    

def eeg_peer_comparison():

    df = pd.DataFrame.from_csv('/home/json/Desktop/peer/fingerprinting_qap.csv')
    df = df[(df.PEER1 <= .2) & (df.TP <=.2)]
    sub_list = df.index.values.tolist()

    def create_sub_list_with_et_and_peer(full_list):

        """Creates a list of subjects with both ET and PEER predictions

        :param full_list: List of subject IDs containing all subjects with at least 2/3 valid calibration scans
        :return: Subject list with both ET and PEER predictions
        """

        et_list = []

        for sub in full_list:

            if (os.path.exists('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')) and \
                    (os.path.exists(
                        '/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')):
                et_list.append(sub)

        return et_list

    et_list = create_sub_list_with_et_and_peer(sub_list)

    with open('/data2/Projects/Lei/Peers/scripts/data_check/TP_bad_sub.txt') as f:
        bad_subs_list_tp = f.readlines()
    with open('/data2/Projects/Lei/Peers/scripts/data_check/DM_bad_sub.txt') as f:
        bad_subs_list_dm = f.readlines()

    bad_subs_list_tp = [x.strip('\n') for x in bad_subs_list_tp]
    bad_subs_list_dm = [x.strip('\n') for x in bad_subs_list_dm]

    et_list = [x for x in et_list if x not in bad_subs_list_tp]
    et_list = [x for x in et_list if x not in bad_subs_list_dm]

    def individual_series(peer_list, et_list):

        et_series = {}
        peer_series = {}

        for sub in peer_list:

            try:

                sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/no_gsr_train1_tp_pred.csv')
                sub_x = sub_df['x_pred']
                sub_y = sub_df['y_pred']

                peer_series[sub] = {'x': sub_x, 'y': sub_y}

            except:

                print('Error with subject ' + sub)

        for sub in et_list:

            try:

                sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
                sub_x = sub_df['x_pred']
                sub_y = sub_df['y_pred']

                et_series[sub] = {'x': sub_x, 'y': sub_y}

            except:

                print('Error with subject ' + sub)

        return et_series, peer_series

    def med_series(peer_list, et_list):
        # peer_list = subjects with valid peer scans
        # et_list = subjects with valid et data

        et_series = {'x': [], 'y': []}
        peer_series = {'x': [], 'y': []}

        for sub in peer_list:

            try:

                sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/no_gsr_train1_tp_pred.csv')
                sub_x = sub_df['x_pred']
                sub_y = sub_df['y_pred']

                if len(sub_x) == 250:
                    peer_series['x'].append(np.array(sub_x))
                    peer_series['y'].append(np.array(sub_y))

                else:
                    print(sub)

            except:

                print('Error with subject ' + sub)

        for sub in et_list:

            try:

                sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
                sub_x = sub_df['x_pred']
                sub_y = sub_df['y_pred']
                et_series['x'].append(np.array(sub_x))
                et_series['y'].append(np.array(sub_y))
            except:

                print('Error with subject ' + sub)

        et_mean_series = {'x': np.nanmedian(et_series['x'], axis=0), 'y': np.nanmedian(et_series['y'], axis=0)}
        peer_mean_series = {'x': np.nanmedian(peer_series['x'], axis=0), 'y': np.nanmedian(peer_series['y'], axis=0)}

        return et_mean_series, peer_mean_series

    et_individual_series, peer_individual_series = individual_series(sub_list, et_list)
    et_mean_series, peer_mean_series = med_series(sub_list, et_list)
    
    pd_et = pd.DataFrame.from_dict(et_mean_series)
    pd_peer = pd.DataFrame.from_dict(peer_mean_series)
    pd_et.to_csv('/home/json/Desktop/peer/et_group_median.csv')
    pd_peer.to_csv('/home/json/Desktop/peer/peer_group_median.csv')

    no_z_x = spearmanr(peer_mean_series['x'], et_mean_series['x'])[0]
    no_z_y = spearmanr(peer_mean_series['y'], et_mean_series['y'])[0]
    
    x_axis = range(len(peer_mean_series['x']))

    plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(4, 8)
    plt.subplot(grid[:2, :])
    plt.title('Median fixation series for ET and PEER')
    plt.plot(x_axis, peer_mean_series['x'], 'r-', label=('PEER, r=' + str(no_z_x)[:5]), alpha=.75)
    plt.plot(x_axis, et_mean_series['x'], 'b-', label='ET', alpha=.75)
    plt.legend(loc=1)
    plt.subplot(grid[2:, :])
    plt.plot(x_axis, peer_mean_series['y'], 'r-', label=('PEER, r=' + str(no_z_y)[:5]), alpha=.75)
    plt.plot(x_axis, et_mean_series['y'], 'b-', label='ET', alpha=.75)
    plt.legend(loc=1)
    plt.savefig('/home/json/Desktop/manuscript_figures/eeg_peer_comp.png', dpi=600)
    plt.show()






###########

def fig_two_old():

    sns.set()

    params = pd.read_csv('/home/json/Desktop/peer/phenotypes_full.csv', index_col='Subject')

    viewtype = 'calibration'
    modality = 'peer'
    cscheme = 'inferno'
    sorted_by = 'age'

    x_stack = []
    y_stack = []

    params = params.sort_values(by=[sorted_by])
    sub_list = params.index.values.tolist()

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}

    fixations = pd.read_csv('stim_vals.csv')
    x_targets = np.tile(np.repeat(np.array(fixations['pos_x']), 5) * monitor_width / 2, 1)
    y_targets = np.tile(np.repeat(np.array(fixations['pos_y']), 5) * monitor_height / 2, 1)
    
    arr = np.zeros((filename_dict[viewtype]['num_vol']))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])

    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(x_targets)
        y_stack.append(y_targets)
        
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)
        y_stack.append(arry)
        
    x_hm = []
    y_hm = []

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)
            
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)
        y_stack.append(arry)            
            
    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(x_targets)
        y_stack.append(y_targets)

    x_hm = np.stack(x_stack)
    y_hm = np.stack(y_stack)

    x_spacing = len(x_hm[0])





    plt.figure(figsize=(10, 10))
    grid = plt.GridSpec(4, 4, hspace=.3)
    plt.subplot(grid[:2, :])
    plt.title('Calibration Scan Predictions')
    ax = sns.heatmap(x_hm, cmap=cscheme, xticklabels=False, yticklabels=False)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    ax.set(ylabel='Subjects')

#    plt.subplot(grid[:2, 2:])
#    plt.title('Calibration Predictions in y-')
#    ax = sns.heatmap(y_hm, cmap=cscheme, xticklabels=False, yticklabels=False)
#    ax.set(xlabel='Volumes', ylabel='Subjects')
#    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
#    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))



    params = pd.read_csv('/home/json/Desktop/peer/phenotypes_full.csv', index_col='Subject')

    viewtype = 'dm'
    modality = 'peer'
    cscheme = 'inferno'
    sorted_by = 'Scan1_meanfd'

    x_stack = []
    y_stack = []

    params = params.sort_values(by=[sorted_by])
    sub_list = params.index.values.tolist()
    
    qap_mri = list(pd.read_csv('/home/json/Desktop/peer/qap_naturalistic_mri.csv').subject)
    
    sub_list = [x for x in sub_list if x in qap_mri]

    filename_dict = {'calibration': {'name': '/gsr0_train1_model_calibration_predictions.csv', 'num_vol': 135},
                     'tp': {'name': '/gsr0_train1_model_tp_predictions.csv', 'num_vol': 250},
                     'dm': {'name': '/gsr0_train1_model_dm_predictions.csv', 'num_vol': 750}}
    
    arr = np.zeros((filename_dict[viewtype]['num_vol']))
    arrx = np.array([-np.round(monitor_width / 2, 0) for x in arr])
    arry = np.array([-np.round(monitor_height / 2, 0) for x in arr])
        
    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)
        y_stack.append(arry)

    for sub in sub_list:

        try:

            if modality == 'peer':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + filename_dict[viewtype]['name'])

            elif modality == 'et':

                temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')

            x_series = list(temp_df['x_pred'])
            y_series = list(temp_df['y_pred'])

            if (len(x_series) == filename_dict[viewtype]['num_vol']) and (len(y_series) == filename_dict[viewtype]['num_vol']):

                x_series = [x if abs(x) < monitor_width/2 + .1*monitor_width else 0 for x in x_series]
                y_series = [x if abs(x) < monitor_height/2 + .1*monitor_height else 0 for x in y_series]

                x_stack.append(x_series)
                y_stack.append(y_series)

        except:

            print('Error processing subject ' + sub)
    
    avg_series_x = np.mean(x_stack, axis=0)
    avg_series_y = np.mean(y_stack, axis=0)

    for num in range(int(np.round(len(sub_list) * .03, 0))):
        x_stack.append(arrx)
        y_stack.append(arry)

    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.insert(0, avg_series_x)
        y_stack.insert(0, avg_series_y)
        
    for num in range(int(np.round(len(sub_list) * .05, 0))):
        x_stack.append(avg_series_x)
        y_stack.append(avg_series_y)

    x_hm = np.stack(x_stack)

    x_spacing = len(x_hm[0])




    plt.subplot(grid[2:, :])
    plt.title('Despicable Me Predictions')
    ax = sns.heatmap(x_hm, cmap=cscheme, xticklabels=False, yticklabels=False)
    ax.set(ylabel='Subjects')
    ax.set(xlabel='Volumes')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
    #plt.savefig('/home/json/Desktop/manuscript_figures/fig2.png', dpi=600)
    plt.show()










    
    
    
    