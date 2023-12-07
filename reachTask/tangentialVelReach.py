#%%

import pandas as pd
import freemocapAnalysis as fm
import matplotlib.pyplot as plt

def plot_tan_vel(file_path):
    fmc = pd.read_csv(file_path, skiprows=range(1, 120))
    data = fm.tanVel(fmc)

    plt.plot(data['time_s'], data['right_wrist_tangential_velocity'])
    plt.xlabel('time_s')
    plt.ylabel('Tangential Velocity')
    plt.title('Tangential Velocity over Time for ' + file_path)
    plt.show()
    return fmc 

# Call the function for each file
fmc1 = plot_tan_vel('recording_11_22_04_gmt-7_by_trajectory.csv')
fmc2 = plot_tan_vel('recording_11_37_04_gmt-7_by_trajectory.csv')
fmc3 = plot_tan_vel('recording_11_52_41_gmt-7_by_trajectory.csv')
# %%
