import pandas as pd
import freemocapAnalysis as fm
import matplotlib.pyplot as plt

def plot_tan_vel(file_path):
    fmc = pd.read_csv(file_path, skiprows=range(1, 120))# nrows=50)
    data = fm.tanVel(fmc)

    plt.plot(data['right_wrist_tangential_velocity'])
    plt.xlabel('Time')
    plt.ylabel('Tangential Velocity')
    plt.title('Tangential Velocity over Time for ' + file_path)
    plt.show()

# Call the function for each file
plot_tan_vel('reachTask/recording_11_22_04_gmt-7_by_trajectory.csv')
plot_tan_vel('reachTask/recording_11_37_04_gmt-7_by_trajectory.csv')
plot_tan_vel('reachTask/recording_11_52_41_gmt-7_by_trajectory.csv')