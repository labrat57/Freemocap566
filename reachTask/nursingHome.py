
#%%
import pandas as pd
import freemocapAnalysis as fm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from freemocapAnalysis import x_y_z_plot
from freemocapAnalysis import animate_3d_plot

fmc = pd.read_csv(r"C:\Users\romyu\OneDrive - University of Calgary\Freemocap 2023\freemocap_data\recording_sessions\session_2023-12-11_11_07_49\recording_11_13_39_gmt-7\recording_11_13_39_gmt-7_by_trajectory.csv")


x_y_z_plot(fmc, 'right_wrist_x', 'right_wrist_y', 'right_wrist_z')

animate_3d_plot(fmc, 'right_wrist_x', 'right_wrist_y', 'right_wrist_z')
# %%
