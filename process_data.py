import numpy as np
import csv
import os


def max_sustained(arr, sustain_samples=3, tolerance=0.5):
    # Take the maximum value within window of length = sustain_samples 
    # with a certain angle tolerance (up and down)
    max_val = None
    for i in range(len(arr) - sustain_samples):
        window = arr[i:i+sustain_samples+1]
        if np.all(np.abs(window - window[0]) < tolerance):
            if (max_val is None) or (window[0] > max_val):
                max_val = window[0]
    if max_val is None:
        return max(arr)
    return max_val


def detect_max_before_fall(angles, fall_threshold=30, window=10):
    # Detect the maximum angle before an abrupt fall using a sliding window
    # fall_threshold: in degrees
    # window: number of samples (time = window / frequency)
    for i in range(window, len(angles)):
        if angles[i - window] - angles[i] > fall_threshold:
            if (i - window) > 0:
                return round(max(angles[:i - window]), 2)
            else:
                return round(max(angles[:i]), 2)
    
    return round(max(angles), 2)


folder_path_1 = './data/elbow_angles/'
folder_path_2 = './data/wrist_angles/'
folder_path_3 = './data/wrist_velocities/'
folder_path_4 = './data/classes/'

elbow_angle_files = []
wrist_angle_files = []
wrist_velocities_files = []
class_files = []
names = []
attempts = []

# Loop to load all the files from the folders
for filename in os.listdir(folder_path_1):
    elbow_angle_files.append(np.load(folder_path_1+filename))

for filename in os.listdir(folder_path_2):
    wrist_angle_files.append(np.load(folder_path_2+filename))

for filename in os.listdir(folder_path_3):
    wrist_velocities_files.append(np.load(folder_path_3+filename))

for filename in os.listdir(folder_path_4):
    class_files.append(np.load(folder_path_4+filename))
    x = filename.split('_')
    names.append(x[1])
    y = x[2] # 1.npy
    attempts.append(y[:-4]) # 1

total_attempts = len(elbow_angle_files)
print(f"Total attempts: {total_attempts}")

data = []
id = 0

for i in range(total_attempts):
    max_flexion_angle = detect_max_before_fall(elbow_angle_files[i])

    data.append({'id': i+1, 'name': names[i], 'attempt': attempts[i], 'max_elbow_flexion_angle [deg]': max_flexion_angle, 
                 'max_elbow_extension_angle [deg]': round(min(elbow_angle_files[i]),2), 'max_wrist_angle [deg]': round(max_sustained(wrist_angle_files[i]),2), 
                 'max_wrist_velocity [m/s]': round(max(wrist_velocities_files[i]),2), 'class': 'in' if class_files[i]==1 else 'out'})


# STORE THE DATA IN A FILE .CSV
with open('dataset.csv', 'a', newline='') as csvfile:
    fieldnames = ['id', 'name', 'attempt', 'max_elbow_flexion_angle [deg]', 'max_elbow_extension_angle [deg]', 
                  'max_wrist_angle [deg]', 'max_wrist_velocity [m/s]', 'class']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # If the file is empty, the header is written
    if csvfile.tell() == 0:  # check if the file is empty
        writer.writeheader()
    # Writes the data
    writer.writerows(data)
