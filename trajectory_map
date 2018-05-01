from heatmappy import Heatmapper
from heatmappy import VideoHeatmapper

import numpy as np
import pandas as pd

from PIL import Image

heatmapper = Heatmapper(
    point_diameter=50,  # the size of each point to be drawn
    point_strength=0.5,  # the strength, between 0 and 1, of each point to be drawn
    opacity=0.95,  # the opacity of the heatmap layer
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

temp_df = pd.DataFrame.from_csv('/data2/Projects/Jake/Human_Brain_Mapping/sub-5343770/gsr0_train1_model_tp_predictions.csv')
x_pred = temp_df['x_pred'][5]
y_pred = temp_df['y_pred'][5]

def x_scale(x_old):
    
    x_new = x_old * 640 / 850 + 640
    
    return x_new
    
def y_scale(y_old):
    
    y_new = y_old * 360 / 525 + 360
    
    return y_new
    
x_pred = x_scale(x_pred)
y_pred = y_scale(y_pred)

example_points = []

for item in range(800):
    example_points.append([x_pred, y_pred, item])
    
heatmap_video = video_heatmapper.heatmap_on_video_path(
    video_path=example_vid,
    points=example_points
)

heatmap_video.write_videofile('/home/json/Desktop/out.mp4', bitrate="5000k", fps=24)