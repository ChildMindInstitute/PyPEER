#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors:
    - Jake Son, 2017-2018  (jake.son@childmind.org)

"""

import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.ticker as ticker

height = 1050
width = 1680

data_path = '/data2/Projects/Jake/Human_Brain_Mapping/'

stim_df = pd.read_csv('/home/json/Desktop/peer/stim_vals.csv')

x_stim = stim_df.pos_x.tolist()
y_stim = stim_df.pos_y.tolist()

x_stim = np.repeat(x_stim, 5) * width/2
y_stim = np.repeat(y_stim, 5) * height/2

###############################################################################
# Distribution of correlation scores

# List of 448 participants used in primary analysis
analysis_df = pd.read_csv('/home/json/Desktop/peer/model_outputs.csv')
sub_list = analysis_df.subject.tolist()

x_corr = []
y_corr = []

for sub in sub_list:
    
    filename = data_path + sub + '/gsr0_train1_model_calibration_predictions.csv'
    
    df = pd.read_csv(filename)
    x_pred = df.x_pred.tolist()
    y_pred = df.y_pred.tolist()
    
    x_corr.append(pearsonr(x_pred, x_stim)[0])
    y_corr.append(pearsonr(y_pred, y_stim)[0])

corr_df = pd.DataFrame.from_dict({'subject': sub_list,
                                  'x_corr': x_corr,
                                  'y_corr': y_corr})

mean_x_corr = np.mean(corr_df.x_corr.tolist()) # .66
mean_y_corr = np.mean(corr_df.y_corr.tolist()) # .58
stdv_x_corr = np.std(corr_df.x_corr.tolist())  # .31
stdv_y_corr = np.std(corr_df.y_corr.tolist())  # .32

###############################################################################

###############################################################################
# Create heatmap for calibration scans

analysis_df = pd.read_csv('/home/json/Desktop/peer/model_outputs.csv')
analysis_df = analysis_df.sort_values(by=['mean_fd'])
sub_list = analysis_df.subject.tolist()

x_stack = []
y_stack = []

for sub in sub_list:
    
    filename = data_path + sub + '/gsr0_train1_model_calibration_predictions.csv'
    
    df = pd.read_csv(filename)
    
    x_series = [x if abs(x) < width/2 + .1*width else 0 for x in df.x_pred.tolist()]
    y_series = [x if abs(x) < height/2 + .1*height else 0 for x in df.y_pred.tolist()]

    x_stack.append(x_series)
    y_stack.append(y_series)

arr = np.zeros(len(x_series))
arrx = np.array([-np.round(width / 2, 0) for x in arr])
arry = np.array([-np.round(height / 2, 0) for x in arr])

for num in range(int(np.round(len(sub_list) * .02, 0))):
    x_stack.append(arrx)
    y_stack.append(arry)

for num in range(int(np.round(len(sub_list) * .02, 0))):
    x_stack.append(x_stim)
    y_stack.append(y_stim)

x_hm = np.stack(x_stack)
y_hm = np.stack(y_stack)

x_spacing = len(x_hm[0])

sns.set()
plt.clf()
ax = sns.heatmap(x_hm)
ax.set(xlabel='Volumes', ylabel='Subjects')
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.round(x_spacing/5, 0)))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
plt.title('Fixation Series for Calibration Scan')
plt.show()



###############################################################################

# Visualization Example

df = pd.read_csv('/data2/Projects/Jake/Human_Brain_Mapping/sub-5343770/gsr0_train1_model_calibration_predictions.csv')

x_pred = df.x_pred.tolist()
y_pred = df.y_pred.tolist()

plt.figure()
plt.subplot(2,1,1)
plt.ylim([-850, 850])
plt.plot(range(len(x_stim)), x_stim, color='black', label='Target')
plt.plot(range(len(x_pred)), x_pred, color='blue', label='PEER')
plt.subplot(2,1,2)
plt.ylim([-600, 600])
plt.plot(range(len(y_stim)), y_stim, color='black', label='Target')
plt.plot(range(len(y_pred)), y_pred, color='blue', label='PEER')
plt.savefig('/home/json/Desktop/fixation_series.png')
plt.show()

pointwise_error = []

for i in range(int(len(x_pred)/5)):
    
    x_delta = x_pred[i] - x_stim[i]
    y_delta = y_pred[i] - y_stim[i]
    pointwise_error.append(np.sqrt((x_delta)**2 + (y_delta)**2))

cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(pointwise_error), vmax=max(pointwise_error))
colors = [cmap(normalize(value)) for value in pointwise_error]

fig, ax = plt.subplots(figsize=(12, 8))
plt.xlim([-width/2, width/2])
plt.ylim([-height/2, height/2])
ax.scatter(x_stim, y_stim, color='black', marker='x')
ax.scatter(x_pred, y_pred, color=colors, alpha=.8)
cax, _ = matplotlib.colorbar.make_axes(ax)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
plt.savefig('/home/json/Desktop/2d_viz.png')
plt.show()


