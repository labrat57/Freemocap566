#%%
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
# then you can do importlib.reload(rf) to reload the module
# Load data
sname = 'ro03_heel' 
datapath = fa.setdatapath("jer") 
fnames = fa.get_list_subject_files(sname,datapath)
pddata = pd.read_csv(fnames[0])
reachr = rf.reachData(pddata)

# %%
time_s = pddata["timestamp"]
time_s = time_s - time_s[0]
time_s = time_s/1e9
lhip = np.array([pddata["left_hip_x"],pddata["left_hip_y"],pddata["left_hip_z"]])
rhip = np.array([pddata["right_hip_x"],pddata["right_hip_y"],pddata["right_hip_z"]])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time_s,lhip.T)
t_samp = [6,16,22]
# find indices closest to each t_samp
ind_samp = [np.argmin(np.abs(time_s - t)) for t in t_samp]
# now plot the points
ax.plot(time_s[ind_samp],lhip.T[ind_samp],'ro')
ax.set_xlabel('time (s)')
ax.set_ylabel('position (m)')
#now show the plot
plt.show()
# %%
pt1 = lhip[:,ind_samp[0]]
pt2 = lhip[:,ind_samp[1]]
pt3 = lhip[:,ind_samp[2]]

# define which 
v1 = pt3 - pt2 # positive right
v2 = pt2 - pt1 # positive forward
# normalize v1 and v2 to be unit length
v1 = v1/np.linalg.norm(v1)
v2 = v2/np.linalg.norm(v2)
# remove any component of v2 that is in the direction of v1
v2 = v2 - np.dot(v2,v1)*v1
v2 = v2/np.linalg.norm(v2)
# now take the cross product
v3 = np.cross(v1,v2)

### compute the rotation matrix to go from the data frame to that defined by v1,v2,v3
R = np.array([v1,v2,v3]).T