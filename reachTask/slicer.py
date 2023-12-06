import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# this opens the files
fileOne = pd.read_csv('reachTask/recording_11_22_04_gmt-7_by_trajectory.csv', skiprows=range(1, 120))# nrows=50)
fileTwo = pd.read_csv('reachTask/recording_11_37_04_gmt-7_by_trajectory.csv', skiprows=range(1, 90))# nrows=50)
fileThree = pd.read_csv('reachTask/recording_11_52_41_gmt-7_by_trajectory.csv', skiprows=range(1, 150))# nrows=50)


def chunk_and_save(filename, base_name):
    # Read the file
    file = pd.read_csv(filename)

    # Calculate the number of chunks
    num_chunks = len(file) // 300

    # Iterate over the DataFrame in chunks of 300 rows
    for i in range(num_chunks):
        # Extract the chunk
        chunk = file.iloc[i*300:(i+1)*300]
        
        # Save the chunk to a .csv file
        chunk.to_csv(f'{base_name}_chunk_{i+1}.csv', index=False)

# Call the function for each file
chunk_and_save('reachTask/recording_11_22_04_gmt-7_by_trajectory.csv', 'fileOne')
chunk_and_save('reachTask/recording_11_37_04_gmt-7_by_trajectory.csv', 'fileTwo')
chunk_and_save('reachTask/recording_11_52_41_gmt-7_by_trajectory.csv', 'fileThree')