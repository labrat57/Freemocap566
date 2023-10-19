import shutil
import os

# Source file path
source_file_path = 'C:\Users\romyu\OneDrive - University of Calgary\freemocap\freemocap_data\recording_sessions\session_2023-10-12_18_05_05\recording_18_09_23_gmt-6\recording_18_09_23_gmt-6_by_trajectory.csv'

# Destination directory path
destination_directory = 'C:\Users\romyu\Freemocap566'

# Copy the file to the destination directory
shutil.copy(source_file_path, destination_directory)

print("File copied successfully.")
