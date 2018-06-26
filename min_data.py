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
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

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

def bootstrap_min_data(sub):
    
    print(str('Begin {}').format(sub))

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
    
    print(('Begin bootstrap for {}').format(sub))
    
    for min_target in range(2, 26):
        
        bootstrap_iteration = 1
    
        while bootstrap_iteration <= 50:
            
            print(str('Target {}, iteration {}').format(min_target, bootstrap_iteration))
            
            if min_target == 25:
                
                bootstrap_iteration = 100
        
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
            
    viz_df = pd.DataFrame.from_dict(bootstrap_dict)
    viz_df.to_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/min_data.csv')
    
    print(str('Completed {}').format(sub))

sub_list = pd.DataFrame.from_csv('/home/json/Desktop/peer/model_outputs.csv').index.tolist()

Parallel(n_jobs=15)(delayed(bootstrap_min_data)(sub)for sub in sub_list)




#####################################################################################################################
# Create DataFrame


min_data_df = {'n_training_points': [], 'x_corr': [], 'y_corr': []}

count = 0

for sub in sub_list:
    
    try:
        
        temp_df = pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/min_data.csv')
        min_data_df['n_training_points'].extend(list(temp_df.n_training_points))
        min_data_df['x_corr'].extend(list(temp_df.x_corr))
        min_data_df['y_corr'].extend(list(temp_df.y_corr))
        
        count += 1
        
    except:
        
        continue

min_data_pd = pd.DataFrame.from_dict(min_data_df)

#####################################################################################################################
# Tukey Plot

sns.set()
sns.boxplot(x='n_training_points', y='x_corr', data=min_data_pd)
plt.show()

#####################################################################################################################
# Minimum Data t-test 

for num in range(2, 25):

    print('Analysis Begin ----------------')
    
    n1_df = min_data_pd[min_data_pd.n_training_points == num][['x_corr', 'y_corr']]
    n2_df = min_data_pd[min_data_pd.n_training_points == num+1][['x_corr', 'y_corr']]

    print(n1_df.shape[0], n2_df.shape[0])
    
    x_s, x_p = ttest_ind(n1_df.x_corr.tolist(), n2_df.x_corr.tolist())
    y_s, y_p = ttest_ind(n1_df.y_corr.tolist(), n2_df.y_corr.tolist())

    print(num, x_p, y_p)


#####################################################################################################################
# Heatmap


n_train = [str(x) for x in np.arange(2, 26)]
df_dict = {'corr': [np.round(x, 2) for x in np.linspace(-1, .95, 40)]}
for item in n_train:
        
    df_dict[item] = np.zeros(40)


df_x = pd.DataFrame.from_dict(df_dict).set_index('corr')
df_y = pd.DataFrame.from_dict(df_dict).set_index('corr')

completed_count = 0

for sub in sub_list[:15]:
    
    try:

        sub_df = pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/min_data.csv')
        
        for row in sub_df.iterrows():
            
            n_targ = str(row[1]['n_training_points'])
            corr_x = float(row[1]['x_corr'])
            corr_y = float(row[1]['y_corr'])
            
            if n_targ == '25':
                
                mult_fact = 50
                
            else:
                
                mult_fact = 1
            
            bin_val_x = -1
            
            while bin_val_x < corr_x - .05:
        
                bin_val_x += .05
                
            bin_val_x = np.round(bin_val_x, 2)
        
            df_x.loc[bin_val_x, n_targ] += 1*mult_fact
            
            bin_val_y = -1
            
            while bin_val_y < corr_x - .05:
        
                bin_val_y += .05
                
            bin_val_y = np.round(bin_val_y, 2)
        
            df_y.loc[bin_val_y, n_targ] += 1*mult_fact
           
        completed_count += 1
        print(str('Completed subject #{}').format(completed_count))
            
    except:
        
        print('Data not yet available')

df_x = df_x.sort_index(ascending=False) / max(df_x.max())
df_y = df_y.sort_index(ascending=False) / max(df_y.max())
df_x = df_x[n_train]
df_y = df_y[n_train]

plt.figure()
sns.heatmap(df_x)
plt.xlabel('Number of Calibration Targets in Training Set')
plt.ylabel('Correlation value bins')
plt.title('Impact of Training Points in Model Accuracy in x-')
plt.show()

sns.heatmap(df_y)
plt.xlabel('Number of Calibration Targets in Training Set')
plt.ylabel('Correlation value bins')
plt.title('Impact of Training Points in Model Accuracy in y-')
plt.show()


























