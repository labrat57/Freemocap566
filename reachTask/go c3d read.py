#%%
import c3d
import numpy as np
import os, sys
import freemocapAnalysis as fa
import reach_fmc as rf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# %%
datadir = fa.setdatapath('jer')

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
puck1 = np.ndarray((npoints,5))
puck2 = np.ndarray((npoints,5))
finger1 = np.ndarray((npoints,5))
wrist1 = np.ndarray((npoints,5))
wrist2 = np.ndarray((npoints,5))
elbow = np.ndarray((npoints,5))
sho  = np.ndarray((npoints,5))

for i, p, analog in reader.read_frames():
  puck1[i-1,:] = p[5,:]
  puck2[i-1,:] = p[6,:]
  finger1[i-1,:] = p[0,:]
  wrist1[i-1,:] = p[1,:]
  wrist2[i-1,:] = p[2,:]
  elbow[i-1,:] = p[3,:]
  sho[i-1,:] = p[4,:]

  
  
  
# %%
