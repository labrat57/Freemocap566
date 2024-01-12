# Run analysis of lin and len datasets.
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
# then you can do importlib.reload(rf) to reload the module

# Load data
sname = 'je'
datapath = fa.setdatapath("jer")
fnames = fa.get_list_subject_files(sname,datapath)
pddata = pd.read_csv(fnames[0])
reachr = rf.reachData(pddata)

reachr.click_add_wrist_starts_ends(numclicks=24,sname=sname)

#%%
plt.plot(reachr.time,reachr.tanvel_wri)
plt.plot(reachr.time[reachr.mov_starts],reachr.tanvel_wri[reachr.mov_starts],'go')
plt.plot(reachr.time[reachr.mov_ends],reachr.tanvel_wri[reachr.mov_ends],'ro')
plt.show()
# %%
