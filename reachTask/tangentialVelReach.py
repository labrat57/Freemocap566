#%%
import freemocapAnalysis as fm
import pandas as pd
data =pd.read_csv('reachTask/recording_11_22_04_gmt-7_by_trajectory.csv', skiprows=range(1, 120))# nrows=50)
data = fm.tanVel(data)