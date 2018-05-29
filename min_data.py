# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:09:13 2018

@author: json
"""

import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
from aux_process import *
from sklearn.svm import SVR
from scipy.stats import pearsonr

monitor_width = 1680
monitor_height = 1050

targets_df = pd.read_csv('/home/json/Desktop/peer/stim_vals.csv')
x_targets = np.array(targets_df.pos_x)
y_targets = np.array(targets_df.pos_y)

x_expected = np.tile(np.repeat(x_targets, 5) * monitor_width / 2, 1)
y_expected = np.tile(np.repeat(y_targets, 5) * monitor_height / 2, 1)

x_targets = np.delete(x_targets, [20, 25]) * monitor_width / 2
y_targets = np.delete(y_targets, [20, 25]) * monitor_height / 2

eye_mask = nib.load('/data2/Projects/Jake/eye_masks/2mm_eye_corrected.nii.gz')
eye_mask = eye_mask.get_data()

sub = 'sub-5343770'

train_scan = nib.load('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/peer1_eyes_sub.nii.gz')
train_scan = train_scan.get_data()
train_scan = np.delete(train_scan, np.concatenate([np.arange(100, 105), np.arange(125, 130)]), axis=3)

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

bootstrap_dict = {'x_corr':[], 'y_corr':[], 'n_training_points': [], 'target_set': []}

for min_target in range(2, 26):
    
    print(str('Beginning bootstrap analysis with {} points').format(min_target))
    print('============================================')
    
    bootstrap_iteration = 1

    while bootstrap_iteration <= 100:
        
        if min_target == 25:
            
            bootstrap_iteration = 100
        
        print(str('Bootstrap iteration {}').format(bootstrap_iteration))
    
        target_array = np.random.choice(25, min_target, replace=False)
        
        target_index = [int(x) for x in np.array([np.linspace(x*5, x*5+4, 5) for x in target_array]).ravel()]
        
        train_sample = train_scan[:, :, :, target_index]
        
        train_data = []
        
        for point in range(min_target):
            
            vol_set = np.arange(point*5, (point+1)*5)
            
            vect = train_sample[:, :, :, vol_set]
            vect = list(np.average(vect, axis=3).ravel())
        
            train_data.append(vect)
        
        x_model = SVR(kernel='linear', C=100, epsilon=.01)
        x_model.fit(train_data, x_targets[target_array])
        y_model = SVR(kernel='linear', C=100, epsilon=.01)
        y_model.fit(train_data, y_targets[target_array])
        
        predicted_x = x_model.predict(test_data)
        predicted_y = y_model.predict(test_data)
        
        predicted_x = np.array([np.round(float(x), 3) for x in predicted_x])
        predicted_y = np.array([np.round(float(x), 3) for x in predicted_y])
        
        x_corr = pearsonr(predicted_x, x_expected)[0]
        y_corr = pearsonr(predicted_y, y_expected)[0]
        
        if ~np.isnan(x_corr) and ~np.isnan(y_corr):
        
            bootstrap_dict['x_corr'].append(x_corr)
            bootstrap_dict['y_corr'].append(y_corr)
            bootstrap_dict['n_training_points'].append(min_target)
            bootstrap_dict['target_set'].append(target_array)
            
            bootstrap_iteration += 1
        
viz_df = df.from_dict(bootstrap_dict)

sns.set()
sns.boxplot(x='n_training_points', y='x_corr', data=viz_df)
plt.show()



