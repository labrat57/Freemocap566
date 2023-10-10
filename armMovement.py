import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('mediapipe_right_hand_3d_xyz.csv')  # opens file

# calculation of time by taking seconds in the clip and dividing by data points given
total_time_seconds = 23
total_rows = 717
time_increment = total_time_seconds / total_rows

# adding a time column to the file
df['Time'] = [i * time_increment for i in range(1, len(df) + 1)]
# adding a velocity column to the file then outputting it
df['speed_right_hand_wrist_x'] = df['right_hand_wrist_x'].diff() / df['Time'].diff()
df.to_csv('output_file.csv', index=False)

# choosing which lines to show in the graph to eliminate noise. roughly seconds 8 to 10. lowering motion
df = pd.read_csv('output_file.csv', skiprows=range(1, 248), nrows=65)

# plotting the graph
plt.plot(df['Time'], df['speed_right_hand_wrist_x'], label='Data')
plt.xlabel('Time(s)')
plt.ylabel('right hand wrist velocity(data/s)')
plt.title('right hand wrist speed')
plt.legend()
plt.show()
