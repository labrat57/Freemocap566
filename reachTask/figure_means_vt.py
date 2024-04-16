#%%
import freemocapAnalysis as fa
import reach_fmc as rf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_formats = ['svg']
# for fitting data to power law
import statsmodels.formula.api as smf

subnames = ['paper01','paper02','paper03','paper07','paper08']

datapath = fa.setdatapath("jer")

cts = np.zeros((len(subnames),7))
ps = np.zeros((len(subnames),7))
#tgts
tgts = np.array([.02,.05,.1,.2,.3,.45,.6])
#define edges intermediate to tgts array in loop
edges_d = np.zeros((tgts.shape[0]+1))
edges_d[0] = 0.0
for i in range(1,len(tgts)):
  edges_d[i] = (tgts[i-1]+tgts[i])/2

edges_d[-1] = tgts[-1]+.15

meanv1 = np.zeros((len(subnames),edges_d.shape[0]))
meanv2 = np.zeros((len(subnames),edges_d.shape[0]))
meanv3 = np.zeros((len(subnames),edges_d.shape[0]))
meanv4 = np.zeros((len(subnames),edges_d.shape[0]))
meanv5 = np.zeros((len(subnames),edges_d.shape[0]))
meanv6 = np.zeros((len(subnames),edges_d.shape[0]))
meanv7 = np.zeros((len(subnames),edges_d.shape[0]))
mlist_vel = list()
mlist_vel.append(meanv1)
mlist_vel.append(meanv2)
mlist_vel.append(meanv3)
mlist_vel.append(meanv4)
mlist_vel.append(meanv5)
mlist_vel.append(meanv6)
mlist_vel.append(meanv7)

meant1 = np.zeros((len(subnames),edges_d.shape[0]))
meant2 = np.zeros((len(subnames),edges_d.shape[0]))
meant3 = np.zeros((len(subnames),edges_d.shape[0]))
meant4 = np.zeros((len(subnames),edges_d.shape[0]))
meant5 = np.zeros((len(subnames),edges_d.shape[0]))
meant6 = np.zeros((len(subnames),edges_d.shape[0]))
meant7 = np.zeros((len(subnames),edges_d.shape[0]))
mlist_t = list()
mlist_t.append(meant1)
mlist_t.append(meant2)
mlist_t.append(meant3)
mlist_t.append(meant4)
mlist_t.append(meant5)
mlist_t.append(meant6)
mlist_t.append(meant7)

meanD1 = np.zeros((len(subnames),edges_d.shape[0]))
meanD2 = np.zeros((len(subnames),edges_d.shape[0]))
meanD3 = np.zeros((len(subnames),edges_d.shape[0]))
meanD4 = np.zeros((len(subnames),edges_d.shape[0]))
meanD5 = np.zeros((len(subnames),edges_d.shape[0]))
meanD6 = np.zeros((len(subnames),edges_d.shape[0]))
meanD7 = np.zeros((len(subnames),edges_d.shape[0]))
mlist_D = list()
mlist_D.append(meanD1)
mlist_D.append(meanD2)
mlist_D.append(meanD3)
mlist_D.append(meanD4)
mlist_D.append(meanD5)
mlist_D.append(meanD6)
mlist_D.append(meanD7)

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
    reachr.click_add_wrist_starts_ends(do_skip_figs=True)
    distancelist, durationlist, peakspeedlist, indlist_middle_mvmnt_start_end = reachr.mainsequence()
    
    # filter out outliers
    peakspeedmax = 1.8 *1000
    durationmax = 2.0
    peakspeedlist[0,peakspeedlist[0,:]>peakspeedmax] = np.nan
    durationlist[0,durationlist[0,:]>durationmax] = np.nan

    # i use a 'mainSeq' object to wrap the D, V, T together. 
    ms = rf.mainSeq(D = distancelist[0,:]/1000, V = peakspeedlist[0,:]/1000,T = durationlist[0,:])
    ct,p,stats,fun_v,fun_t = rf.fit_ct(ms,normT = 1.0, normV = .5)
    ps[isub,ind_cond] = p
    cts[isub,ind_cond] = ct

    # discretize distancelist 
    digitized = np.digitize(distancelist[0,:]/1000,edges_d)
    mlist_vel[ind_cond][isub,:] = np.array([np.mean(peakspeedlist[0,digitized==i])/1000 for i in range(1,edges_d.shape[0]+1)])
    mlist_t[ind_cond][isub,:] = np.array([np.mean(durationlist[0,digitized==i]) for i in range(1,edges_d.shape[0]+1)])
    mlist_D[ind_cond][isub,:] = np.array([np.mean(distancelist[0,digitized==i]/1000) for i in range(1,edges_d.shape[0]+1)])

    # plot the main sequence data, then the fit on top.
    dline = np.linspace(0,0.7,100)
    ax[0,col_plt].plot(ms.D,ms.V,'o',color=c)
    ax[0,col_plt].plot(dline,fun_v(dline,ct),'-',color=c)
    ax[1,col_plt].plot(ms.D,ms.T,'o',color=c)
    ax[1,col_plt].plot(dline,fun_t(dline,ct),'-',color=c)

    # set the limits. 
    # if there are outliers, this will be a potential problem.
    # ax[0,col_plt].set_xlim([0,.8])
    # ax[0,col_plt].set_ylim([0, 2])
    # ax[1,col_plt].set_xlim([0,.8])
    # ax[1,col_plt].set_ylim([0, 2])
    
    ax[0,col_plt].set_xlabel('Distance (m)')
    ax[0,col_plt].set_ylabel('Peak speed (m/s)')
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

# display cts.
np.set_printoptions(suppress=True)
print(np.round(cts.T,2))

#%%
# in mlist_vel, mlist_t, and mlist_D, set any 0 values to None
for i in range(7):
  mlist_vel[i][mlist_vel[i]==0] = None
  mlist_t[i][mlist_t[i]==0] = None
  mlist_D[i][mlist_D[i]==0] = None

# %% plot error bars from mlist_vel
conditionlist = ['p','p2','s','f','t','m','l']
# where in the list conditionlist is 'f'?

inds = [fa.index_from_condition(c) for c in conditionlist]
f1,ax1 = plt.subplots(1,2,figsize=(10,5))

for i in range(4):
  N = len(subnames)
  c = fa.color_from_condition(conditionlist[i])
  D = mlist_D[i]
  V = mlist_vel[i]
  T = mlist_t[i]
  ax1[0].errorbar(np.nanmean(D,axis=0),np.nanmean(V,axis=0),yerr=np.nanstd(V,axis=0),color=c,linestyle='none',capsize=3,marker='o')
  ax1[1].errorbar(np.nanmean(D,axis=0),np.nanmean(T,axis=0),yerr=np.nanstd(T,axis=0),color=c,linestyle='none',capsize=3,marker='o')

  ms = rf.mainSeq(D = np.nanmean(D,axis=0), V = np.nanmean(V,axis=0),T = np.nanmean(T,axis=0))

  A = np.nanmean(D,axis=0)**(3/4)
  b = np.nanmean(V,axis=0)

  results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
  Kv      = results.params.A # note: this Kv is independently fit, until new gains for movement are generated.

  A = np.nanmean(D,axis=0)**(1/4)
  b = np.nanmean(T,axis=0)
  results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
  Kt      = results.params.A # note: this Kt is independently fit, as above.

  dline = np.concatenate((np.linspace(0,0.02,100),np.linspace(0.03,.75,100)))
  V_star = Kv*dline**(3/4)
  T_star = Kt*dline**(1/4)
  ax1[0].plot(dline,V_star,'-',color=c)
  ax1[1].plot(dline,T_star,'-',color=c)


ax1[0].set_ylim([0,2])
ax1[0].set_xlim([0,.4])
ax1[0].set_ylabel('Peak speed (m/s)')
ax1[0].set_xlabel('Distance (m)')

ax1[1].set_ylim([0,1.6])
ax1[1].set_xlim([0,.4])
ax1[1].set_ylabel('Duration (s)')
ax1[1].set_xlabel('Distance (m)')

# remove top and right box lines
ax1[0].spines['top'].set_visible(False)
ax1[0].spines['right'].set_visible(False)
ax1[1].spines['top'].set_visible(False)
ax1[1].spines['right'].set_visible(False)

for label in (ax1[0].get_xticklabels() + ax1[0].get_yticklabels()+ax1[1].get_xticklabels()+ax1[1].get_yticklabels()):
  label.set_fontsize(18)
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels

ax1[0].set_xticks([0,.1,.2,.3,.4])
ax1[0].set_yticks([0,.5,1,1.5,2])
ax1[1].set_xticks([0,.1,.2,.3,.4])
ax1[1].set_yticks([0,.5,1,1.5,2])


# set dimensions of the subplots 
f1.tight_layout()
f1.savefig('figures/means_vt.pdf')

plt.show()

# %%
f2,ax2 = plt.subplots(1,2,figsize=(10,5))
for i in range(4,7):
  N = len(subnames)
  c = fa.color_from_condition(conditionlist[i])
  D = mlist_D[i]
  V = mlist_vel[i]
  T = mlist_t[i]
  ax2[0].errorbar(np.nanmean(D,axis=0),np.nanmean(V,axis=0),yerr=np.nanstd(V,axis=0),color=c,linestyle='none',capsize=3,marker='o')
  ax2[1].errorbar(np.nanmean(D,axis=0),np.nanmean(T,axis=0),yerr=np.nanstd(T,axis=0),color=c,linestyle='none',capsize=3,marker='o')

  ms = rf.mainSeq(D = np.nanmean(D,axis=0), V = np.nanmean(V,axis=0),T = np.nanmean(T,axis=0))

  A = np.nanmean(D,axis=0)**(3/4)
  b = np.nanmean(V,axis=0)

  results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
  Kv      = results.params.A # note: this Kv is independently fit, until new gains for movement are generated.

  A = np.nanmean(D,axis=0)**(1/4)
  b = np.nanmean(T,axis=0)
  results = smf.ols('b ~ A -1', data = pd.DataFrame({'A':A,'b':b})).fit()
  Kt      = results.params.A # note: this Kt is independently fit, as above.

  dline = np.concatenate((np.linspace(0,0.02,100),np.linspace(0.03,.75,100)))
  V_star = Kv*dline**(3/4)
  T_star = Kt*dline**(1/4)
  ax2[0].plot(dline,V_star,'-',color=c)
  ax2[1].plot(dline,T_star,'-',color=c)


ax2[0].set_ylim([0,2])
ax2[0].set_xlim([0,.4])
ax2[0].set_ylabel('Peak speed (m/s)')
ax2[0].set_xlabel('Distance (m)')

ax2[1].set_ylim([0,1.6])
ax2[1].set_xlim([0,.4])
ax2[1].set_ylabel('Duration (s)')
ax2[1].set_xlabel('Distance (m)')

# remove top and right box lines
ax2[0].spines['top'].set_visible(False)
ax2[0].spines['right'].set_visible(False)
ax2[1].spines['top'].set_visible(False)
ax2[1].spines['right'].set_visible(False)

for label in (ax2[0].get_xticklabels() + ax2[0].get_yticklabels()+ax2[1].get_xticklabels()+ax2[1].get_yticklabels()):
  label.set_fontsize(18)
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels

ax2[0].set_xticks([0,.2,.4,.6,.8])
ax2[0].set_yticks([0,.5,1,1.5,2])
ax2[1].set_xticks([0,.2,.4,.6,.8])
ax2[1].set_yticks([0,.5,1,1.5,2])


# set dimensions of the subplots 
f2.tight_layout()
f2.savefig('figures/means_acc_vt.pdf')

plt.show()

# %%
