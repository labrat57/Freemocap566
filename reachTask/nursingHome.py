
#%%
import sys, os                                # for reading files and using directories properly.
import pandas as pd                           # for dataframes
import freemocapAnalysis as fm                # for analysis functions
import matplotlib.pyplot as plt               # for plotting
import numpy as np                           # for math
import matplotlib.animation as animation     # for animation
# from freemocapAnalysis import x_y_z_plot      # we imported this above.
# from freemocapAnalysis import animate_3d_plot # we imported this above.

#setup paths
fm.addReachPath()
datapath = fm.setdatapath("jeremy")

# look at a particular nursing home recording. 
name_session = "session_2023-12-11_11_07_49"
name_recording = "recording_11_13_39_gmt-7"
name_file = "recording_11_13_39_gmt-7_by_trajectory.csv"
fname_full = os.path.join(datapath, name_session,name_recording, name_file)
print(fname_full)
data = pd.read_csv(fname_full)
data = fm.tan_vel(data)

fm.x_y_z_plot(data, 'right_wrist_x', 'right_wrist_y', 'right_wrist_z')

#fm.animate_3d_plot(fmc, 'right_wrist_x', 'right_wrist_y', 'right_wrist_z')

#%% 
right = fm.reachData(data)
#work with the data to snip out the desired moments in time. 

#%%
