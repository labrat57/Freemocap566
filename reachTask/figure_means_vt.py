#%%
import freemocapAnalysis as fa
import reach_fmc as rf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

subnames = ['paper01','paper02','paper03','paper07','paper08']

datapath = fa.setdatapath("jer")

cts = np.zeros((len(subnames),7))
ps = np.zeros((len(subnames),7))
for isub, subj in enumerate(subnames):
  files = fa.get_list_subject_files(subj,datapath)
  conds = fa.get_list_conditions(subj)
  f,ax = plt.subplots(2,2)
  for ifl,fpath in enumerate(files):
    cond = conds[ifl]
    c = fa.color_from_condition(cond)
    i = fa.index_from_condition(cond)
    col_plt = 0
    if (i>2) :
      col_plt = 1

    pddata            = pd.read_csv(fpath)
    reachr            = rf.reachData(pddata,datapath)
    reachr.fraw_name  = os.path.basename(fpath)
    reachr.R          = fa.get_cached_R(subj)
    reachr.click_add_wrist_starts_ends(do_skip_figs=True)
    distancelist, durationlist, peakspeedlist, indlist_middle_mvmnt_start_end = reachr.mainsequence()
    
    # i use a 'mainSeq' object to wrap the D, V, T together. 
    ms = rf.mainSeq(D = distancelist[0,:]/1000, V = peakspeedlist[0,:]/1000,T = durationlist[0,:])
    ct,p,stats,fun_v,fun_t = rf.fit_ct(ms)
    ps[isub,i] = p
    cts[isub,i] = ct

    # plot the main sequence data, then the fit on top.
    dline = np.linspace(0,0.7,100)
    ax[0,col_plt].plot(ms.D,ms.V,'o',color=c)
    ax[0,col_plt].plot(dline,fun_v(dline,ct),'-',color=c)
    ax[1,col_plt].plot(ms.D,ms.T,'o',color=c)
    ax[1,col_plt].plot(dline,fun_t(dline,ct),'-',color=c)

    ax[0,col_plt].set_xlim([0,.8])
    ax[0,col_plt].set_ylim([0, 2])
    ax[0,col_plt].set_xlabel('Distance (m)')
    ax[0,col_plt].set_ylabel('Peak speed (m/s)')

    ax[1,col_plt].set_xlim([0,.8])
    ax[1,col_plt].set_ylim([0, 2])
    ax[1,col_plt].set_xlabel('Distance (m)')
    ax[1,col_plt].set_ylabel('Duration (s)')
    f.suptitle(subj)
    # remote top and right box lines
    ax[0,col_plt].spines['top'].set_visible(False)
    ax[0,col_plt].spines['right'].set_visible(False)
    ax[1,col_plt].spines['top'].set_visible(False)
    ax[1,col_plt].spines['right'].set_visible(False)
  
  # tight layout
  f.tight_layout()
  f.show()

# %%
np.set_printoptions(suppress=True)
np.round(cts.T,2)
# %%
