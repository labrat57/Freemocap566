import pandas as pd
import numpy as np
import math


def find_3d_coordinate(file_name, row_number):
    df = pd.read_csv(file_name)
    x, y = df['X'].values[row_number], df['Y'].values[row_number]
    vector = np.array([x, y, 0])
    return vector


# data for the first set of coordinates
vector1 = find_3d_coordinate('click_coordinates_x.csv', 0)
vector2 = find_3d_coordinate('click_coordinates_y.csv', 0)
vector3 = find_3d_coordinate('click_coordinates_z.csv', 0)

# data for the second set of coordinates
vector4 = find_3d_coordinate('click_coordinates_x.csv', 1)
vector5 = find_3d_coordinate('click_coordinates_y.csv', 1)
vector6 = find_3d_coordinate('click_coordinates_z.csv', 1)


# Vector definitions
coordinate1 = [vector1[1], vector2[1], vector3[1]]
coordinate2 = [vector4[1], vector5[1], vector6[1]]

# Magnitude of both vectors
magnitude1 = math.sqrt(sum([x**2 for x in coordinate1]))
print("coordinate1 magnitude: ", magnitude1)

magnitude2 = math.sqrt(sum([x**2 for x in coordinate2]))
print("coordinate2 magnitude: ", magnitude2)

# Distance between both points
distance_between_points = math.sqrt((vector4[1] - vector1[1])**2 + (vector5[1] - vector2[1])**2 + (vector6[1] - vector3[1])**2)
print("distance between points: ", distance_between_points)

