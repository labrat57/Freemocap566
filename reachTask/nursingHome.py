#%%
import sys, os                                # for reading files and using directories properly.
import pandas as pd                           # for dataframes
import freemocapAnalysis as fm                # for analysis functions
import reach_fmc as rf
import matplotlib
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
datapd = pd.read_csv(fname_full)
datapd = fm.tan_vel(datapd)

fm.x_y_z_plot(datapd, 'right_wrist_x', 'right_wrist_y', 'right_wrist_z')
#fm.animate_3d_plot(fmc, 'right_wrist_x', 'right_wrist_y', 'right_wrist_z')

right = rf.reachData(datapd)

#%%
#work with the data to snip out the desired moments in time. 
starts,ends = right.click_add_starts_ends(right.time,right.wri_f,24)
# %%
fig,ax=plt.subplots(4,1)
ax[0].plot(right.time, right.wri_f[0,:])
ax[1].plot(right.time, right.wri_f[1,:])
ax[2].plot(right.time, right.wri_f[2,:])
ax[3].plot(right.time, right.tanvelwri)
ax[0].plot(right.time[right.mov_starts], right.wri[0,right.mov_ends], 'ro')

# #%%
# # Compute the delta time for each reach
# delta_time = right.time[right.mov_ends] - right.time[right.mov_starts]

# # Compute the peak tanvel between each start-end pair
# peak_tanvel = []
# for i in range(len(right.mov_starts)):
#   peak_tanvel.append(np.max(right.tanvelwri[right.mov_starts[i]:right.mov_ends[i]]))

# # Print the results
# print("Delta Time for each reach:", delta_time)
# print("Peak Tanvel between each start-end pair:", peak_tanvel)

# %%
