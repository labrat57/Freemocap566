# Freemocap566
Freemocap Project
Python Version
(3.8.2)

Dependencies:
- matplotlib
- pandas
- numpy
- from scipy.signal import butter, filtfilt

Log - code for Jer to run
2023-12-06 reachTask\tangentialVelReach.py
            this code now works to get the TanVel of the reaching data i collected on monday. The output is in m/s
            also added the freemocapAnalysis.py file for all the functionc from now on
2023-11-30 tanVel.py
                This code calculates the tangential velocity of the FreeMoCap data and the EMC data

2023-11-29: EMCanimation and FMCanimation
                the files for EMC are already in the folder ready to run
                The fmc animation file is called recording_13_58_54... and is also in the folder already. if you want to run more freemocap files, youll have to call those in.


the data from comapring the freemocap (fmc) to the expensive mocap (emc) is found in the "comparison" folder.

The EMCsorting.py file converts the output emc files to something more usable for us. ive already converted all the data. the converted data is called "filteredEMCData" these file are also copied to the shared folder on the cloud.

the "FMCanimation" file will animate all of our freemocap data. i will modify it a bit more for the rest of our files. currently it only works for 1 file.

the "EMCanimation" file works for the emc data that has been filtered. the graph takes time to run and doesnt move very quickly so be patient with it. i think the scaling needs some work or something like that. 


the "graphTest" file is not important

next up i will generalize the fmc animation and run more tests on the emc animation
i would also like to clean up the fmc animation file with some functions.