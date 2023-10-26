import pandas as pd
import numpy as np


def find_3d_coordinate(file_name, row_number):
    df = pd.read_csv(file_name)
    x, y = df['X'].values[row_number], df['Y'].values[row_number]
    vector = np.array([x, y, 0])
    return vector


# Read the data for the first set of coordinates
vector1 = find_3d_coordinate('click_coordinates_x.csv', 2)
vector2 = find_3d_coordinate('click_coordinates_y.csv', 2)
vector3 = find_3d_coordinate('click_coordinates_z.csv', 2)

# Compute the 3D coordinate for the first set of coordinates
new_coordinate1 = np.array([0, 0, 0]) + (vector1 - vector2) + (vector1 - vector3)

# Read the data for the second set of coordinates
vector4 = find_3d_coordinate('click_coordinates_x.csv', 3)
vector5 = find_3d_coordinate('click_coordinates_y.csv', 3)
vector6 = find_3d_coordinate('click_coordinates_z.csv', 3)

# Compute the 3D coordinate for the second set of coordinates
new_coordinate2 = np.array([0, 0, 0]) + (vector4 - vector5) + (vector4 - vector6)

# absolute difference between the two coordinates
absolute_difference = np.abs(new_coordinate1 - new_coordinate2)

# Print the results
print(f"The 3D coordinate for the first set of coordinates is: {tuple(new_coordinate1)}")
print(f"The 3D coordinate for the second set of coordinates is: {tuple(new_coordinate2)}")
print(f"The difference between the two coordinates is: {tuple(absolute_difference)}")
