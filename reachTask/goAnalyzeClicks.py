# Run analysis of lin and len datasets.
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
# then you can do importlib.reload(rf) to reload the module

# Load data
sname = 'ro' # at the moment, 'je' and 'ro' are the only clean datasets that i can tell. and ro is short. 
datapath = fa.setdatapath("rom") 
fnames = fa.get_list_subject_files(sname,datapath)
pddata = pd.read_csv(fnames[0])
reachr = rf.reachData(pddata)

#%% this uses romeo's click detection. It will check first if there is a click file, and if not, it will create one.
reachr.click_add_wrist_starts_ends(numclicks=24,sname=sname)

#%% plot the clicks.
plt.plot(reachr.time,reachr.tanvel_wri)
plt.plot(reachr.time[reachr.mov_starts],reachr.tanvel_wri[reachr.mov_starts],'go')
plt.plot(reachr.time[reachr.mov_ends],reachr.tanvel_wri[reachr.mov_ends],'ro')
plt.show()
# %% Here you could extract peak tanvel.
# in reachr, the field is reachr.tanvel_wri.
# for all of the fields in reachr, see reach_fmc.py, lines 15-33.
