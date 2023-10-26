import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
df = pd.read_csv('filtered_right_wrist.csv')#, skiprows=range(1, 400), nrows=600)

# column data
x = df['right_wrist_x']
y = df['right_wrist_y']
z = df['right_wrist_z']

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, c='r')#, marker='o')


# Set labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Plot')

# Show the plot
plt.show()
