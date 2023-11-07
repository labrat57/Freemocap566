import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('wrist_movement.csv', skiprows=range(1, 820), nrows=50)  # opens file with only 1 good point of data

# 1455 rows
total_time_seconds = 51
total_rows = 1455
time_increment = total_time_seconds / total_rows

# adding a time column to the file
df['Time'] = [i * time_increment for i in range(1, len(df) + 1)]
# adding a velocity column to the file then outputting it
# df['speed_right_hand_x'] = df[body_column].diff() / df['Time'].diff()
# df.to_csv('output_wrist_file.csv', index=False)

# x data

plt.plot(df['right_wrist_x'], label='Data')
plt.xlabel('Time(s)')
plt.ylabel('wrist position (mm)')
plt.title('wrist position')
plt.legend()
# Initialize the list to store the coordinates
coordinates_x = []


# Function to capture mouse clicks
def onclick_x(event):
    global coordinates_x
    coordinates_x.append((event.xdata, event.ydata))
    df_x = pd.DataFrame(coordinates_x, columns=['X', 'Y'])
    df_x.to_csv('click_coordinates_x.csv', index=False)
    print(f"Coordinates after {len(coordinates_x)} clicks saved to 'click_coordinates_x.csv'")


# Connect the click event to the function
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick_x)

plt.show()

# y data
plt.plot(df['right_wrist_y'], label='Data')
plt.xlabel('Time(s)')
plt.ylabel('wrist position (mm)')
plt.title('wrist position')
plt.legend()
# Initialize the list to store the coordinates
coordinates_y = []


# Function to capture mouse clicks
def onclick_y(event):
    global coordinates_y
    coordinates_y.append((event.xdata, event.ydata))
    df_y = pd.DataFrame(coordinates_y, columns=['X', 'Y'])
    df_y.to_csv('click_coordinates_y.csv', index=False)
    print(f"Coordinates after {len(coordinates_y)} clicks saved to 'click_coordinates_y.csv'")


# Connect the click event to the function
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick_y)

plt.show()

# z data
plt.plot(df['right_wrist_z'], label='Data')
plt.xlabel('Time(s)')
plt.ylabel('wrist position (mm)')
plt.title('wrist position')
plt.legend()
# Initialize the list to store the coordinates
coordinates_z = []


# Function to capture mouse clicks
def onclick_z(event):
    global coordinates_z
    coordinates_z.append((event.xdata, event.ydata))
    df_z = pd.DataFrame(coordinates_z, columns=['X', 'Y'])
    df_z.to_csv('click_coordinates_z.csv', index=False)
    print(f"Coordinates after {len(coordinates_z)} clicks saved to 'click_coordinates_z.csv'")


# Connect the click event to the function
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick_z)

plt.show()

# using the data in the 'click_coordinates_x,y,z' files, i want to know the distance between these points using pythagorean theorem.
