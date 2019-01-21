#!/usr/bin/env python
"""
Script used on the command line to estimate eye movements

Authors:
    - Jake Son, 2017-2018  (jake.son@childmind.org)

"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVR
from sklearn.externals import joblib
import time
from peer_func import *

if __name__ == "__main__":

    project_dir, top_data_dir, stimulus_path = scaffolding()

    os.chdir(project_dir)

    for i, dataset in enumerate([x for x in os.listdir(top_data_dir) if not x.startswith('.')]):

        data_dir = os.path.abspath(os.path.join(top_data_dir, dataset))

        output_dir = os.path.abspath(os.path.join(data_dir, 'outputs'))

        print(('\nPredicting fixations for participant #{}').format(i+1))
        print('====================================================')

        configs = load_config()

        filepath = os.path.join(data_dir, configs['test_file'])

        print('\nLoad Data')
        print('====================================================')

        eye_mask_path = configs['eye_mask_path']
        eye_mask = nib.load(eye_mask_path).get_data()

        data = load_data(filepath)

        for vol in range(data.shape[3]):
            output = np.multiply(eye_mask, data[:, :, :, vol])
            data[:, :, :, vol] = output

        volumes = data.shape[3]

        start_time = time.time()
        mean_data = np.mean(data, axis=3)
        std_data = np.std(data, axis=3)
        std_data[std_data == 0] = 1

        for i in range(volumes):
            data[:, :, :, i] = (data[:, :, :, i]-mean_data)/std_data
        elapsed_time = time.time() - start_time
        print("Elapsed time: " + str(elapsed_time))

        if int(configs['use_gsr']):

            print('\nGlobal Signal Regression')
            print('====================================================')

            eye_mask_path = configs['eye_mask_path']
            data = global_signal_regression(data, eye_mask_path)

        raveled_data = [data[:, :, :, vol].ravel() for vol in np.arange(data.shape[3])]

        xmodel, ymodel, xmodel_name, ymodel_name = load_model(output_dir)

        print('\nPredicting Fixations')
        print('====================================================')

        print('Fixations saved to specified output directory.')

        x_fix, y_fix = predict_fixations(xmodel, ymodel, raveled_data)

        x_fixname, y_fixname = save_fixations(x_fix, y_fix, xmodel_name, ymodel_name, output_dir)

        print('\nEstimating Eye Movements')
        print('====================================================')

        estimate_em(x_fix, y_fix, x_fixname, y_fixname, output_dir)

        print('Eye movements saved to specified output directory.')

    print('\n')
