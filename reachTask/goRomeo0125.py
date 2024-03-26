#%%
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
# then you can do importlib.reload(rf) to reload the module
# Load data
sname = 'ro_0125_Y' # at the moment, 'je' and 'ro' are the only clean datasets that i can tell. and ro is short. 
datapath = fa.setdatapath("jer") 
fnames = fa.get_list_subject_files(sname,datapath)
pddata = pd.read_csv(fnames[0])
reachr = rf.reachData(pddata)

plt.plot(reachr.time,reachr.tanvel_wri)
# %%
reachr.click_add_wrist_starts_ends(numclicks=24,sname=sname)

#%% using reachr.mov_ends and mov_starts, plot the extracted data of the wrist, in reachr.wri_f
fig,ax=plt.subplots(4,1)
cutreaches = list()
for imov in range(len(reachr.mov_starts)):
  inds = np.arange(reachr.mov_starts[imov],reachr.mov_ends[imov])
  tzeroed = reachr.time[inds] - reachr.time[reachr.mov_starts[imov]]
  ax[0].plot(tzeroed,reachr.wri_f[0,inds])
  ax[1].plot(tzeroed,reachr.wri_f[1,inds])
  ax[2].plot(tzeroed,reachr.wri_f[2,inds])
  ax[3].plot(tzeroed,reachr.tanvel_wri[inds])
  cutreaches.append(np.array((tzeroed,reachr.wri_f[:,inds])))
fig.show()
  
# %%
