# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:09:28 2018

@author: json
"""

import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
from aux_process import *
from sklearn.svm import SVR
from scipy.stats import pearsonr

monitor_width = 1680
monitor_height = 1050

targets_df = pd.read_csv('/home/json/Desktop/peer/stim_vals.csv')
x_targets = np.array(targets_df.pos_x) * monitor_width / 2
y_targets = np.array(targets_df.pos_y) * monitor_height / 2
x_expected = np.tile(np.repeat(x_targets, 5), 1)
y_expected = np.tile(np.repeat(y_targets, 5), 1)

cpac_dir = '/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/output/pipeline_RU_CPAC/'

eye_mask = nib.load('/data2/Projects/Jake/eye_masks/2mm_eye_corrected.nii.gz')
eye_mask = eye_mask.get_data()

sub_list = pd.DataFrame.from_csv('/home/json/Desktop/peer/phenotypes_full.csv').index.tolist()

censor_dict = {'x_censor': [], 'y_censor': [], 'x_raw': [], 'y_raw': []}

df_sub_list = []
df_x_corr = []
df_y_corr = []

for sub in sub_list:
    
    print(('Volume censoring for {}').format(sub))

    train_fd = np.loadtxt(cpac_dir + sub + '_ses-1/frame_wise_displacement/_scan_peer_run-1/FD.1D')
    
    censor_index = []
    
    for vol in range(train_fd.shape[0]):
        
        if train_fd[vol] > .2:
            
            censor_index.append(vol)
    
    train_scan = nib.load('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/peer1_eyes_sub.nii.gz')
    train_scan = train_scan.get_data()
    
    test_scan = nib.load('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/peer2_eyes_sub.nii.gz')
    test_scan = test_scan.get_data()
    
    for vol in range(train_scan.shape[3]):
        
        masked = np.multiply(eye_mask, train_scan[:, :, :, vol])
        train_scan[:, :, :, vol] = masked
        
    for vol in range(test_scan.shape[3]):
        
        masked = np.multiply(eye_mask, test_scan[:, :, :, vol])
        test_scan[:, :, :, vol] = masked
    
    train_scan = mean_center_var_norm(train_scan)
    test_scan = mean_center_var_norm(test_scan)
    
    test_data = []
    
    for vol in range(135):
    
        vect = test_scan[:, :, :, vol]
        vect = list(np.ravel(vect))
        
        test_data.append(vect)
    
    train_data = []
    removed_index = []
    
    for num in range(27):
        
        vol_set = [x for x in np.arange(num*5, (num+1)*5) if x not in censor_index]
        
        if len(vol_set) != 0:
        
            vect = list(np.average(train_scan[:, :, :, vol_set], axis=3).ravel())
            train_data.append(vect)
            
        else:
            
            removed_index.append(num)
    
    x_targets_sub = list(np.delete(np.array(x_targets), removed_index))
    y_targets_sub = list(np.delete(np.array(y_targets), removed_index))
    
    x_model = SVR(kernel='linear', C=100, epsilon=.01)
    x_model.fit(train_data, x_targets_sub)
    y_model = SVR(kernel='linear', C=100, epsilon=.01)
    y_model.fit(train_data, y_targets_sub)
    
    predicted_x = x_model.predict(test_data)
    predicted_y = y_model.predict(test_data)
    
    x_corr_censor = pearsonr(predicted_x, x_expected)[0]
    y_corr_censor = pearsonr(predicted_y, y_expected)[0]
    
    temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_parameters.csv')
    x_corr = temp_df.corr_x[0]
    y_corr = temp_df.corr_y[0]
    
    df_sub_list.append(sub)
    df_x_corr.append(x_corr_censor)
    df_y_corr.append(y_corr_censor)

#    if ~np.isnan(x_corr_censor) and ~np.isnan(y_corr_censor) and ~np.isnan(x_corr) and ~np.isnan(y_corr):
#
#        censor_dict['x_censor'].append(x_corr_censor)
#        censor_dict['y_censor'].append(y_corr_censor)
#        censor_dict['x_raw'].append(x_corr)
#        censor_dict['y_raw'].append(y_corr)
#        
#censor_df = pd.DataFrame.from_dict(censor_dict)

sns.set()
sns.regplot(x='x_censor', y='x_raw', data=censor_df)
plt.title('Effect of Volume Censoring on PEER1_r in x-')
plt.xlim([-.5, 1])
plt.ylim([-.5, 1])
plt.show()










