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
  #print('frame {}: point {}, analog {}'.format(i, p.shape, analog.shape))
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


#%%
import c3d
import numpy as np
import os, sys
import freemocapAnalysis as fa
import reach_fmc as rf
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

datadir = fa.setdatapath('rom')
fname = os.path.join(datadir,'PhaseSpace-2024-01-25','trial3.c3d')

reader = c3d.Reader(open(fname, 'rb'))
npoints = 0
for i, p, analog in reader.read_frames():
  #print('frame {}: point {}, analog {}'.format(i, p.shape, analog.shape))
  npoints = i
  p.shape

puck1 = np.ndarray((npoints,5))

for i, p, analog in reader.read_frames():
  puck1[i-1,:] = p[5,:]

# magnitude of the position vector of puck1
puck1_magnitude = np.sqrt(puck1[:,0]**2 + puck1[:,1]**2 + puck1[:,2]**2)

collection_frequency = 960.00
time_duration = npoints / collection_frequency

# time array for plotting
time_array = np.linspace(0, time_duration, npoints)

# magnitude over time
plt.plot(time_array, puck1_magnitude)  
plt.xlabel('Time (seconds)')
plt.ylabel('Puck1 Position Magnitude')
plt.title('Puck1 Position Magnitude over Time')
plt.show()

#%%

def tan_vel(data, collection_frequency):
    #velocity in each dimension
    velocity_x = np.gradient(data[:,0], 1/collection_frequency)
    velocity_y = np.gradient(data[:,1], 1/collection_frequency)
    velocity_z = np.gradient(data[:,2], 1/collection_frequency)

    # Calculate tanvel
    tangential_velocity = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)

    return tangential_velocity

# example of how to call: puck1_tan_vel = tan_vel(puck1, collection_frequency)

#%%
#magnitude of the position vector of puck1
puck1_magnitude = np.sqrt(puck1[:,0]**2 + puck1[:,1]**2 + puck1[:,2]**2)

#tangential velocity of puck1
puck1_tan_vel = tan_vel(puck1, collection_frequency)

plt.plot(puck1_magnitude, puck1_tan_vel)
plt.xlabel('Displacement')
plt.ylabel('Tangential Velocity')
plt.title('Tangential Velocity vs Displacement Puck1')
plt.show()

#%%
#%%
# this will be the freemocap code that i use to get plots of the data from the same trial.



# %%
