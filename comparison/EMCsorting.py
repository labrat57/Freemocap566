# this file sorts the C3D files into the same format as the freemocap data file for easy manipulation
#columns will be: body_part_x, body_part_y, body_part_z, and so on. The body parts will have to be indentified by the user

# remove the first few lines of code as they are for calibration
# add the code to a csv file
import pandas as pd

skip_rows = [0] + list(range(2, 19))
emc = pd.read_csv(r"C:\Users\romyu\OneDrive - University of Calgary\Freemocap 2023\freemocap_data\recording_sessions\Calibrate wrist\calibrationOfFreemocap_WristOnly_001.csv", skiprows=skip_rows)

# look at the rows of emc data. if the has a 1 in the id marker column then it gets moved to a new dataframe with the same name as the id marker

num_points = len(emc.index)

# Initialize an empty 3D list
three_d_list = []

def process_id(id_num):
    # Initialize an empty DataFrame
    filtered_emc = pd.DataFrame()

    for i in range(len(emc.index)):
        if emc.loc[i, ' id'] == id_num:
            # Convert the Series to a DataFrame and concatenate
            filtered_emc = pd.concat([filtered_emc, pd.DataFrame([{f'{id_num}_pos_x': emc.loc[i, ' pos_x'], f'{id_num}_pos_y': emc.loc[i, ' pos_y'], f'{id_num}_pos_z': emc.loc[i, ' pos_z']}])], ignore_index=True)

    return filtered_emc

# Initialize an empty DataFrame to hold all the data
all_data = pd.DataFrame()

# Call the function for each id from 1 to 10 and concatenate the results
for id_num in range(0, 17):
    all_data = pd.concat([all_data, process_id(id_num)], axis=1)

# Write the all_data DataFrame to a CSV file

# change the name of this to have it write to a different file (004,5,6...)
all_data.to_csv('comparison/filteredEMCData_wristOnly_001.csv', index=False)

