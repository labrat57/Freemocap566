#%%
import pandas as pd
import matplotlib.pyplot as plt
from freemocapAnalysis import pos_time_plot, tan_vel, click_starts_ends
from freemocapAnalysis import get_list_subject_files , setdatapath
from reach_fmc import __init__

import freemocapAnalysis as fa
import reach_fmc as rf
import numpy as np


#1) distance; 2) duration and 3) peak speed. it looks like for these plots it should be easy to pick out the middle spead of those triplets, the middle being the peak speed. 
#just clikc the middle peak and we will get speed and duration. for distance use pytag to get the distance(using wehre you cick as the start adn end of the vector
#I'd guess duration is easy to calculate from the two minima surrounding the max. I assume that's when the hand slows to pick up the object.
datadir = setdatapath('rom')


# Load the click data
trial_1_clicks = pd.read_csv('C:/code/Freemocap566/processed_clicks/hpl_trial1_savedclicks.csv')
trial_2_clicks = pd.read_csv('C:/code/Freemocap566/processed_clicks/hpl_trial2_savedclicks.csv')
trial_3_clicks = pd.read_csv('C:/code/Freemocap566/processed_clicks/hpl_trial3_savedclicks.csv')

trial1_filenamelist = get_list_subject_files('hpl_trial1', datadir)
trial2_filenamelist = get_list_subject_files('hpl_trial2', datadir)
trial3_filenamelist = get_list_subject_files('hpl_trial3', datadir)

# Create reachData objects
pdf = pd.read_csv(trial1_filenamelist[0])
reachr_trial1 = rf.reachData(pdf)
pdf = pd.read_csv(trial2_filenamelist[0])
reachr_trial2 = rf.reachData(pdf)
pdf = pd.read_csv(trial3_filenamelist[0])
reachr_trial3 = rf.reachData(pdf)

#%%
plt.plot(reachr_trial1.time,reachr_trial1.wri_f[0,:])
plt.show()

#%%

# Define the trials and their corresponding click data
subjects = ['trial_1', 'trial_2', 'trial_3'] #subjects to load data for
clicks = {'trial_1': trial_1_clicks, 'trial_2': trial_2_clicks, 'trial_3': trial_3_clicks} # clicks for each subject


# Load the data for each trial
data = {}
for sname in subjects:
    # Load data
    str_who = 'rom'  # or 'jer', depending on which user's data path you want
    datapath = setdatapath(str_who)
    fnames = get_list_subject_files(sname, datapath)
    subject_data = [pd.read_csv(fname) for fname in fnames]

    # Calculate the tangential velocity for each DataFrame
    for df in subject_data:
        side = 'right'
        body_parts = ['wrist', 'elbow', 'shoulder']  # we only want wrist anyways
        df = tan_vel(df, side, body_parts)
    
    data[sname] = subject_data

# this will be the distance plot
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

# plot the time it takes for each pair of indexes
# Create a list to store the durations for each action
durations = []

for sname, subject_data in data.items():
    # Loop over each pair of start and end indices in the clicks DataFrame for the current subject
    for i in range(0, len(clicks[sname]), 2):
        # Get the start and end indices for the current action
        start, end = clicks[sname].iloc[i, 0], clicks[sname].iloc[i+1, 0]
        for df in subject_data:
            # Slice the DataFrame to only include the rows for the current action
            slice_df = df.loc[start:end]
            # Check if slice_df is not empty
            if not slice_df.empty:
                # Calculate the duration
                duration = slice_df['timestamp'].iloc[-1] - slice_df['timestamp'].iloc[0]
                # Add the duration to the list
                durations.append(duration)

# Plot the durations for the current subject
plt.plot(range(1, len(durations) + 1), durations, label=f'{sname}')

# Duration plot
plt.xlabel('Action')
plt.ylabel('Duration')
plt.legend()
plt.show()


# find the max value in each pair, plot these as the peak speeds
# Create a list to store the peak velocities for each action
peak_velocities = []

for sname, subject_data in data.items():
    # Loop over each pair of start and end indices in the clicks DataFrame for the current subject
    for i in range(0, len(clicks[sname]), 2):
        # Get the start and end indices for the current action
        start, end = clicks[sname].iloc[i, 0], clicks[sname].iloc[i+1, 0]
        for df in subject_data:
            # Slice the DataFrame to only include the rows for the current action
            slice_df = df.loc[start:end]
            # Check if slice_df is not empty
            if not slice_df.empty:
                # Create a reach_fmc object
                reach_data = rf.reachData(slice_df)
                # Calculate the peak velocity
                peak_velocity = np.max(reach_data.tanvel_wri)
                # Add the peak velocity to the list
                peak_velocities.append(peak_velocity)

# Plot the peak velocities for the current subject
plt.plot(range(1, len(peak_velocities) + 1), peak_velocities, label=f'{sname}')

# Peak velocity plot
plt.xlabel('Action')
plt.ylabel('Peak Velocity')
plt.legend()
plt.show()
