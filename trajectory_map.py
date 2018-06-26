from heatmappy import Heatmapper
from heatmappy import VideoHeatmapper

import os
import numpy as np
import pandas as pd

from PIL import Image
from joblib import Parallel, delayed

heatmapper = Heatmapper(
    point_diameter=50,  # the size of each point to be drawn
    point_strength=0.5,  # the strength, between 0 and 1, of each point to be drawn
    opacity=0.5,  # the opacity of the heatmap layer
    colours='default',  # 'default' or 'reveal'
                        # OR a matplotlib LinearSegmentedColorMap object 
                        # OR the path to a horizontal scale image
    grey_heatmapper='PIL'  # The object responsible for drawing the points
                           # Pillow used by default, 'PySide' option available if installed
)

video_heatmapper = VideoHeatmapper(
    heatmapper  # the img heatmapper to use (like the heatmapper above, for example)
)

example_vid = '/data2/Projects/Lei/Peers/The_Present_Seg/Clips/video11.mp4'
example_vid = '/home/json/Desktop/tp_aud_removed_full.mp4'
example_vid = '/home/json/Desktop/PEERS_trimed.mp4'

#####

def load_data(min_scan=2):

    """Returns list of subjects with at least the specified number of calibration scans

    :param min_scan: Minimum number of scans required to be included in subject list
    :return: Dataframe containing subject IDs, site of MRI collection, number of calibration scans, and motion measures
    :return: List containing subjects with at least min_scan calibration scans
    """

    params = pd.read_csv('/home/json/Desktop/peer/model_outputs.csv', index_col='subject', dtype=object)
    params = params.convert_objects(convert_numeric=True)

    if min_scan == 2:

        params = params[(params.scan_count == 2) | (params.scan_count == 3)]

    elif min_scan == 3:

        params = params[params.scan_count == 3]

    sub_list = params.index.values.tolist()

    return params, sub_list

params, sub_list = load_data(min_scan=2)

params = params.sort_values(by='mean_fd')
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

def mean_series(peer_list, et_list):
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

    et_mean_series = {'x': np.nanmean(et_series['x'], axis=0), 'y': np.nanmean(et_series['y'], axis=0)}
    peer_mean_series = {'x': np.nanmean(peer_series['x'], axis=0), 'y': np.nanmean(peer_series['y'], axis=0)}
    
    return et_mean_series, peer_mean_series
    
et_mean_series, peer_mean_series = mean_series(sub_list, sub_list)
    
fixations = []

for sub in sub_list[:25]:

    try:
    
#        sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_tp_predictions.csv')
#        sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/et_device_pred.csv')
        sub_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/' + sub + '/gsr0_train1_model_calibration_predictions.csv')
        x_pred = sub_df['x_pred']
        y_pred = sub_df['y_pred']
        
#        x_pred = x_scale(x_pred)
#        y_pred = y_scale(y_pred)
        
        x_pred = x_pred * 1440/840  # For calibration heatmap
        y_pred = y_pred * 900/525   # For calibration heatmap
        
        x_pred = [x+1440 for x in x_pred]
        y_pred = [x+900 for x in y_pred]
        
        for num1 in range(135):
            
            for num2 in range(800*num1, 800*(num1+1)):
                
                if num2 %50 == 0:
                    
                    fixations.append([x_pred[num1], y_pred[num1], num2])
                    
        print('Subject ' + sub.strip('sub-') + ' completed.')
            
    except:
        
        print('Error with subject ' + sub)
    
heatmap_video = video_heatmapper.heatmap_on_video_path(
    video_path=example_vid,
    points=fixations
)

heatmap_video.write_videofile('/home/json/Desktop/calibration_heatmap.mp4')