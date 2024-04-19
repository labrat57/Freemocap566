#%%
import c3d
import numpy as np
import os, sys
import freemocapAnalysis as fa
import reach_fmc as rf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# %%
datadir = fa.setdatapath('rom')

# %%
fname = os.path.join(datadir,'PhaseSpace-2024-01-25','trial3.c3d')
# %%
# class c3d_reach():
#     ifinger = 0
#     iwri1   = 1
#     iwri2   = 2
#     ielb = 3
#     isho = 4
#     ipuc1 = 5
#     ipuc2 = 6
#%%
reader = c3d.Reader(open(fname, 'rb'))
npoints = 0
for i, p, analog in reader.read_frames():
  print('frame {}: point {}, analog {}'.format(i, p.shape, analog.shape))
  npoints = i
  p.shape

# %%
puck1 = np.ndarray((npoints,3))
puck2 = np.ndarray((npoints,3))
finger1 = np.ndarray((npoints,3))
wrist1 = np.ndarray((npoints,3))
wrist2 = np.ndarray((npoints,3))
elbow = np.ndarray((npoints,3))
sho  = np.ndarray((npoints,3))

for i, p, analog in reader.read_frames():
  puck1[i-1,:] = p[5,0:3]
  puck2[i-1,:] = p[6,0:3]
  finger1[i-1,:] = p[0,0:3]
  wrist1[i-1,:] = p[1,0:3]
  wrist2[i-1,:] = p[2,0:3]
  elbow[i-1,:] = p[3,0:3]
  sho[i-1,:] = p[4,0:3]

  
#%%
fs = 480  # Sample frequency in Hz
time_s = np.arange(wrist1.shape[0]) / fs

  
# %%
#initialize tv with zeros
tanvel_puck = np.zeros((npoints,1))
# list comprehension to compute the three vels with gradient
tanvel_puck = np.sqrt(np.sum(np.gradient(puck1,axis=0)**2,axis=1))
tanvel_wrist2 = np.sqrt(np.sum(np.gradient(wrist2,axis=0)**2,axis=1))

#%%
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time_s, tanvel_puck,linewidth=3)
ax.plot(time_s, tanvel_wrist2)

ax.set_ylim([0,2])
fig.show()

# %%