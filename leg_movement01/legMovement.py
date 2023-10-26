import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('lower_body_data - Sheet1.csv')  # opens file


# 697 rows
total_time_seconds = 24
total_rows = 697
time_increment = total_time_seconds / total_rows

# adding a time column to the file
df['Time'] = [i * time_increment for i in range(1, len(df) + 1)]
# adding a velocity column to the file then outputting it
df['speed_left_ankle_x'] = df['left_ankle_x'].diff() / df['Time'].diff()
df.to_csv('output_leg_file.csv', index=False)

df = pd.read_csv('output_leg_file.csv', skiprows=range(1, 60), nrows=350)

# plotting the graph
plt.plot(df['Time'], df['speed_left_ankle_x'], label='Data')
plt.xlabel('Time(s)')
plt.ylabel('leg velocity(mm/s)')
plt.title('ankle acceleration')
plt.legend()
plt.show()

# i want another plot but this time it will be the distance traveled by the ankle
#i can use the data given to me to calculate the distance traveled by the ankle

plt.plot(df['Time'], df['left_ankle_x'], label='Data')
plt.xlabel('Time(s)')
plt.ylabel('leg position (mm)')
plt.title('ankle speed')
plt.legend()
plt.show()
