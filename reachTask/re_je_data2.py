# this file we will plot the following things
# max tangential velocities between each movement start and movement end pair
#the displacment between each movement start and movement end pair (using the wri_r fields)
#then plot against each other with x being the displacement and y being the max tangential velocity
#%%
import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from freemocapAnalysis import get_list_subject_files , tan_vel, setdatapath
from reach_fmc import __init__



# indeces we need
je_clicks = pd.read_csv('C:/code/Freemocap566/processed_clicks/je_savedclicks.csv')
ro_clicks = pd.read_csv('C:/code/Freemocap566/processed_clicks/ro_savedclicks.csv')



subjects = ['ro', 'je']  # subjects to load data for
clicks = {'ro': ro_clicks, 'je': je_clicks}  # clicks for each subject
data = {}

for sname in subjects:
    # Load data
    str_who = 'rom'  # or 'jer', depending on which user's data path you want
    datapath = setdatapath(str_who)
    fnames = get_list_subject_files(sname, datapath)
    subject_data = [pd.read_csv(fname) for fname in fnames]
    sub_reach_data = [rf.reachData(df) for df in subject_data]  
    # Calculate the tangential velocity for each DataFrame
    for df in subject_data:
        side = 'right'
        body_parts = ['wrist', 'elbow', 'shoulder']  # we only want wrist anyways
        df = tan_vel(df, side, body_parts)
    
    data[sname] = subject_data

# Loop over each subject and their data
for sname, subject_data in data.items():
    # Loop over each pair of start and end indices in the clicks DataFrame for the current subject
    for i in range(0, len(clicks[sname]), 2):
        # Get the start and end indices for the current action
        start, end = clicks[sname].iloc[i, 0], clicks[sname].iloc[i+1, 0]
        # Loop over each DataFrame in the subject's data
        for df in subject_data:
            # Slice the DataFrame to only include the rows for the current action
            slice_df = df.loc[start:end]
            # Plot the tangential velocity for the current action
            plt.plot(slice_df.index, slice_df['right_wrist_tangential_velocity'], label=f'{sname} Action {i//2+1}')# last bit here is just the labeling

# tangential vel plot
plt.xlabel('Index')
plt.ylabel('Tangential Velocity')
#plt.legend()
plt.show()


# Loop over each subject and their data
for sname, subject_data in data.items():
    # Create a list to store the displacements for each action
    displacements = []
    # Loop over each pair of start and end indices in the clicks DataFrame for the current subject
    for i in range(0, len(clicks[sname]), 2):
        # Get the start and end indices for the current action
        start, end = clicks[sname].iloc[i, 0], clicks[sname].iloc[i+1, 0]
        for df in subject_data:
            # Slice the DataFrame to only include the rows for the current action
            slice_df = df.loc[start:end]
            # Check if slice_df is not empty
            if not slice_df.empty:
                # Calculate the displacement
                displacement = np.sqrt(
                    (slice_df['right_wrist_x'].iloc[-1] - slice_df['right_wrist_x'].iloc[0])**2 +
                    (slice_df['right_wrist_y'].iloc[-1] - slice_df['right_wrist_y'].iloc[0])**2 +
                    (slice_df['right_wrist_z'].iloc[-1] - slice_df['right_wrist_z'].iloc[0])**2
                )
                # Add the displacement to the list - this is to make it a line plot and not a scatter plot
                displacements.append(displacement)
    # Plot the displacements for the current subject
    plt.plot(range(1, len(displacements) + 1), displacements, label=f'{sname}')

# Displacement plot
plt.xlabel('Action')
plt.ylabel('Displacement')
plt.legend()
plt.show()


