import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('wrist_movement.csv')  # opens file

body_column = 'right_wrist_x'
# options: 'right_index_x', 'right_thumb_x', 'right_wrist_x'

# 1455 rows
total_time_seconds = 51
total_rows = 1455
time_increment = total_time_seconds / total_rows

# adding a time column to the file
df['Time'] = [i * time_increment for i in range(1, len(df) + 1)]
# adding a velocity column to the file then outputting it
df['speed_right_hand_x'] = df[body_column].diff() / df['Time'].diff()
df.to_csv('output_wrist_file.csv', index=False)

df = pd.read_csv('output_wrist_file.csv', skiprows=range(1, 820), nrows=50)
# want time from 16 seconds onwards
# plotting the graph
#plt.plot(df['Time'], df['speed_right_hand_x'], label='Data')
#plt.xlabel('Time(s)')
#plt.ylabel('arm velocity(mm/s)')
#plt.title('wrist acceleration')
#plt.legend()
#plt.show()

# i want another plot but this time it will be the distance traveled by the ankle
# add a line along the x axis at 5000mm

plt.plot(df['Time'], df[body_column], label='Data')
plt.xlabel('Time(s)')
plt.ylabel('wrist position (mm)')
plt.title('wrist position')
#plt.axhline(y=-1800, color='r', linestyle='-')
#plt.axhline(y=-2300, color='r', linestyle='-')
plt.legend()
# Initialize the list to store the coordinates
coordinates = []

# Function to capture mouse clicks
def onclick(event):
    global coordinates
    coordinates.append((event.xdata, event.ydata))
    df = pd.DataFrame(coordinates, columns=['X', 'Y'])
    df.to_csv('click_coordinates.csv', index=False)
    print(f"Coordinates after {len(coordinates)} clicks saved to 'click_coordinates.csv'")

# Connect the click event to the function
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

plt.show()
