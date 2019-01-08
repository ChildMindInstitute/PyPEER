#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors:
    - Jake Son, 2017-2018  (jake.son@childmind.org)

"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

height = 1050
width = 1680

stim_df = pd.read_csv('/home/json/Desktop/peer/stim_vals.csv')

x_stim = stim_df.pos_x.tolist()
y_stim = stim_df.pos_y.tolist()

x_stim = np.repeat(x_stim, 5) * width/2
y_stim = np.repeat(y_stim, 5) * height/2

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


