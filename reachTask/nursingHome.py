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
fnames = fm.get_list_subject_files('lin',datapath)
fname_full = fnames[0]
datapd = pd.read_csv(fname_full)
datapd = fm.tan_vel(datapd)

fm.x_y_z_plot(datapd, 'right_wrist_x', 'right_wrist_y', 'right_wrist_z')
#fm.animate_3d_plot(fmc, 'right_wrist_x', 'right_wrist_y', 'right_wrist_z')

right = rf.reachData(datapd)
starts,ends = right.click_add_starts_ends(right.time,right.wri_f,24)

# %%
fig,ax=plt.subplots(4,1)
ax[0].plot(right.time, right.wri_f[0,:])
ax[1].plot(right.time, right.wri_f[1,:])
ax[2].plot(right.time, right.wri_f[2,:])
ax[3].plot(right.time, right.tanvelwri)
ax[0].plot(right.time[starts], right.wri[0,ends], 'ro')


#%% add velocity
right.wri_ddt = fm.vel3D(right.time, right.wri_f)

plt.plot(right.time, right.wri_ddt[0,:]/1000)
plt.xlabel('time (s)')
plt.ylabel('velocity (m/s)')

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
