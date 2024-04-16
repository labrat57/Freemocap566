#%%
import freemocapAnalysis as fa
import reach_fmc as rf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import statsmodels.formula.api as smf

subnames = ['paper02']

datapath = fa.setdatapath("jer")

for isub, subj in enumerate(subnames):
  files = fa.get_list_subject_files(subj,datapath)
  conds = fa.get_list_conditions(subj)
  f,ax = plt.subplots(2,2)
  for ifl,fpath in enumerate(files):
    cond = conds[ifl]
    c         = fa.color_from_condition(cond)
    ind_cond  = fa.index_from_condition(cond)
    
    # set the column for plotting. we plot p, p2, slow, fast all on one plot; t m b on another. 
    # these conditions are ordered: p, p2, slow, fast, t, m, b
    col_plt = 0
    if (ind_cond>fa.index_from_condition('f')) : # we plot p, p2, slow, fast all on one plot; t m b on another. 
      col_plt = 1

    pddata            = pd.read_csv(fpath)
    reachr            = rf.reachData(pddata,datapath)
    reachr.fraw_name  = os.path.basename(fpath)
    reachr.R          = fa.get_cached_R(subj)
    reachr.click_add_wrist_starts_ends(do_skip_figs=False)
    distancelist, durationlist, peakspeedlist, indlist_middle_mvmnt_start_end = reachr.mainsequence()
    
    # i use a 'mainSeq' object to wrap the D, V, T together. 
    ms = rf.mainSeq(D = distancelist[0,:]/1000, V = peakspeedlist[0,:]/1000,T = durationlist[0,:])
    ct,p,stats,fun_v,fun_t = rf.fit_ct(ms,normT = 1.0, normV = .3)

    f = plt.figure()
    ax = f.add_subplot(111,projection='3d')
    for im in range(0,len(reachr.mov_ends)):
      # plot sh elb wri data. 
      indrange = np.arange(reachr.mov_starts[im],reachr.mov_ends[im])
      wri_r = reachr.R @ reachr.wri_f[:,indrange]
      elb_r = reachr.R @ reachr.elb_f[:,indrange]
      sho_r = reachr.R @ reachr.sho_f[:,indrange]
              
      # plot in 3D
      wri_r = wri_r/1000
      sho_r = sho_r/1000
      elb_r = elb_r/1000

      sho_fix = sho_r[:,0:1]

      wri_z = wri_r - sho_fix
      elb_z = elb_r - sho_fix
      sh_rz = sho_r - sho_fix

      ax.plot(wri_z[0,:],wri_z[1,:],wri_z[2,:])

      if im ==0:

        ax.plot(np.concatenate((wri_z[0,0:1],elb_z[0,0:1],sh_rz[0,0:1])),
                  np.concatenate((wri_z[1,0:1],elb_z[1,0:1],sh_rz[1,0:1])),
                  np.concatenate((wri_z[2,0:1],elb_z[2,0:1],sh_rz[2,0:1])),c='k',linewidth=3)
          
        sh_l = reachr.R @ np.array([pddata['left_shoulder_x'][indrange[0:1]],pddata['left_shoulder_y'][indrange[0:1]],pddata['left_shoulder_z'][indrange[0:1]]])
        hi_r = reachr.R @ np.array([pddata['right_hip_x'][indrange[0:1]],pddata['right_hip_y'][indrange[0:1]],pddata['right_hip_z'][indrange[0:1]]])
        hi_l = reachr.R @ np.array([pddata['left_hip_x'][indrange[0:1]],pddata['left_hip_y'][indrange[0:1]],pddata['left_hip_z'][indrange[0:1]]])

        sh_l = sh_l/1000
        hi_r = hi_r/1000
        hi_l = hi_l/1000

        sh_lz = sh_l - sho_fix
        hi_rz = hi_r - sho_fix
        hi_lz = hi_l - sho_fix

        ax.plot(np.concatenate((sh_lz[0,0:1],sh_rz[0,0:1],hi_rz[0,0:1],hi_lz[0,0:1],sh_lz[0,0:1])),
                np.concatenate((sh_lz[1,0:1],sh_rz[1,0:1],hi_rz[1,0:1],hi_lz[1,0:1],sh_lz[1,0:1])),
                np.concatenate((sh_lz[2,0:1],sh_rz[2,0:1],hi_rz[2,0:1],hi_lz[2,0:1],sh_lz[2,0:1])),c='k',linewidth=3)

        el_l = reachr.R @ np.array([pddata['left_elbow_x'][indrange[0:1]],pddata['left_elbow_y'][indrange[0:1]],pddata['left_elbow_z'][indrange[0:1]]])
        wr_l = reachr.R @ np.array([pddata['left_wrist_x'][indrange[0:1]],pddata['left_wrist_y'][indrange[0:1]],pddata['left_wrist_z'][indrange[0:1]]])        

        el_l = el_l/1000
        wr_l = wr_l/1000

        el_lz = el_l - sho_fix
        wr_lz = wr_l - sho_fix

        ax.plot(np.concatenate((sh_lz[0,0:1],el_lz[0,0:1],wr_lz[0,0:1])),
                np.concatenate((sh_lz[1,0:1],el_lz[1,0:1],wr_lz[1,0:1])),
                np.concatenate((sh_lz[2,0:1],el_lz[2,0:1],wr_lz[2,0:1])),c='k',linewidth=3)
      
    
    ax.set_xlim([-.7,.7])
    ax.set_ylim([-.1,.9])
    ax.set_zlim([-.6,.1])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_aspect('equal')
    ax.view_init(90,-90,0)
    f.show()
    # use os to get basename
    fbase = os.path.basename(fpath)
    fsave = os.path.join('reachTask','figures',fbase[:-4]+'_3d.pdf')
    f.savefig(fsave)
    input('press enter')
# %%
