import matplotlib.pyplot as plt
import numpy as np
import os

folder_path_1 = './plot/filtered/'
folder_path_2 = './plot/interpolated/'
folder_path_3 = './plot/results/'

a_arm_filtered, a_forearm_filtered, a_hand_filtered = [], [], []
w_arm_filtered, w_forearm_filtered, w_hand_filtered = [], [], []
a_arm_interp, a_forearm_interp, a_hand_interp = [], [], []
w_arm_interp, w_forearm_interp, w_hand_interp = [], [], []
elbow_angles, wrist_angles = [], []
arm_velocities, forearm_velocities, hand_velocities = [], [], []
total_speed_arm, total_speed_forearm, total_speed_hand = [], [], []

name = ''
attempts = []
total_files = 0

print("Write the name of the person (no uppercase letter): ")
while True:
    name = input()
    
    for filename in os.listdir(folder_path_1):
        x = filename.split('_')
        if x[2] == name and x[1] == 'arm':
            if x[0] == 'a':
                a_arm_filtered.append(np.load(folder_path_1+filename))
                total_files += 1
                y = x[-1]
                attempts.append(y[:-4])
            else:
                w_arm_filtered.append(np.load(folder_path_1+filename))
                total_files += 1
        elif x[2] == name and x[1] == 'forearm':
            if x[0] == 'a':
                a_forearm_filtered.append(np.load(folder_path_1+filename))
                total_files += 1
            else:
                w_forearm_filtered.append(np.load(folder_path_1+filename))
                total_files += 1
        elif x[2] == name and x[1] == 'hand':
            if x[0] == 'a':
                a_hand_filtered.append(np.load(folder_path_1+filename))
                total_files += 1
            else:
                w_hand_filtered.append(np.load(folder_path_1+filename))
                total_files += 1

    for filename in os.listdir(folder_path_2):
        x = filename.split('_')
        if x[3] == name and x[1] == 'arm':
            if x[0] == 'a':
                a_arm_interp.append(np.load(folder_path_2+filename))
                total_files += 1
            else:
                w_arm_interp.append(np.load(folder_path_2+filename))
                total_files += 1
        elif x[3] == name and x[1] == 'forearm':
            if x[0] == 'a':
                a_forearm_interp.append(np.load(folder_path_2+filename))
                total_files += 1
            else:
                w_forearm_interp.append(np.load(folder_path_2+filename))
                total_files += 1
        elif x[3] == name and x[1] == 'hand':
            if x[0] == 'a':
                a_hand_interp.append(np.load(folder_path_2+filename))
                total_files += 1
            else:
                w_hand_interp.append(np.load(folder_path_2+filename))
                total_files += 1

    for filename in os.listdir(folder_path_3):
        x = filename.split('_')
        if x[2] == name and x[1] == 'ang':
            if x[0] == 'elb':
                elbow_angles.append(np.load(folder_path_3+filename))
                total_files += 1
            else:
                wrist_angles.append(np.load(folder_path_3+filename))
                total_files += 1
        elif x[2] == name and x[0] == 'vel':
            if x[1] == 'arm':
                arm_velocities.append(np.load(folder_path_3+filename))
                total_files += 1
            elif x[1] == 'forearm':
                forearm_velocities.append(np.load(folder_path_3+filename))
                total_files += 1
            else:
                hand_velocities.append(np.load(folder_path_3+filename))
                total_files += 1
        elif x[3] == name and x[0] == 'tot':
            if x[2] == 'arm':
                total_speed_arm.append(np.load(folder_path_3+filename))
                total_files += 1
            elif x[2] == 'forearm':
                total_speed_forearm.append(np.load(folder_path_3+filename))
                total_files += 1
            else:
                total_speed_hand.append(np.load(folder_path_3+filename))
                total_files += 1  

    if total_files != 0:
        print("\nOK\n")
        break 
    else:
        print("Invalid name. Please try again.\n")
        print("Write the name of the person (no uppercase letter): ")


# PLOT THE FILTERED DATA
accel_range = 8*9.81    # ±8g para el acelerómetro (to m/s2)
gyro_range = 1000* (np.pi/180)    # ±1000 dps para el giroscopio (to rad/s)

for i in range(len(attempts)):
    plt.figure()
    plt.suptitle(f"{name.capitalize()}, attempt {attempts[i]}", fontsize=13, fontweight="bold")

    plt.subplot(2,1,1)
    plt.plot(a_arm_filtered[i], label=["X axis", "Y axis", "Z axis"])
    # Reference lines
    plt.axhline(y=accel_range, color='r', linestyle='--', label=f'Máx ±{accel_range/9.81}g')
    plt.axhline(y=-accel_range, color='r', linestyle='--')
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel('Acceleration (m/s²)', fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Raw Acc Arm", fontsize=11)

    plt.subplot(2,1,2)
    plt.plot(w_arm_filtered[i], label=["X axis", "Y axis", "Z axis"])
    # Reference lines
    plt.axhline(y=gyro_range, color='r', linestyle='--', label=f'Máx ±{round((gyro_range/(np.pi/180)),1)} dps')
    plt.axhline(y=-gyro_range, color='r', linestyle='--')
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel('Angular Velocity (rad/s)', fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Raw Gyro Arm", fontsize=11)

    plt.subplots_adjust(hspace=0.3)


# PLOT THE FILTERED VS INTERPOLATED DATA
for i in range(len(attempts)):
    plt.figure()
    plt.suptitle(f"{name.capitalize()}, attempt {attempts[i]}", fontsize=13, fontweight="bold")

    plt.subplot(3,1,1)
    plt.plot(w_arm_filtered[i], label=["Filtered (X)", "Filtered (Y)", "Filtered (Z)"], linestyle="dotted")
    plt.plot(w_arm_interp[i], label=["Interpolated (X)", "Interpolated (Y)", "Interpolated (Z)"])
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel('Angular Velocity (rad/s)', fontsize=9)
    plt.legend(ncol=2, loc="upper right", fontsize=8)
    plt.title("Gyro Arm Comparison", fontsize=11)

    plt.subplot(3,1,2)
    plt.plot(w_forearm_filtered[i], label=["Filtered (X)", "Filtered (Y)", "RFilteredaw (Z)"], linestyle="dotted")
    plt.plot(w_forearm_interp[i], label=["Interpolated (X)", "Interpolated (Y)", "Interpolated (Z)"])
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel('Angular Velocity (rad/s)', fontsize=9)
    plt.legend(ncol=2, loc="upper right", fontsize=8)
    plt.title("Gyro Forearm Comparison", fontsize=11)

    plt.subplot(3,1,3)
    plt.plot(w_hand_filtered[i], label=["Filtered (X)", "Filtered (Y)", "Filtered (Z)"], linestyle="dotted")
    plt.plot(w_hand_interp[i], label=["Interpolated (X)", "Interpolated (Y)", "Interpolated (Z)"])
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel('Angular Velocity (rad/s)', fontsize=9)
    plt.legend(ncol=2, loc="upper right", fontsize=8)
    plt.title("Gyro Hand Comparison", fontsize=11)

    plt.subplots_adjust(hspace=0.5)

# ----------------------------------

for i in range(len(attempts)):
    plt.figure()
    plt.suptitle(f"{name.capitalize()}, attempt {attempts[i]}", fontsize=13, fontweight="bold")

    a = a_arm_interp[i]
    f = a_forearm_interp[i]
    h = a_hand_interp[i]

    plt.subplot(3,1,1)
    plt.plot(a[:,0], label="Arm")
    plt.plot(f[:,0], label="Forearm")
    plt.plot(h[:,0], label="Hand")
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel('Acceleration (m/s²)', fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Interpolated Accelerometer Data (X axis)", fontsize=11)

    plt.subplot(3,1,2)
    plt.plot(a[:,1], label="Arm")
    plt.plot(f[:,1], label="Forearm")
    plt.plot(h[:,1], label="Hand")
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel('Acceleration (m/s²)', fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Interpolated Accelerometer Data (Y axis)", fontsize=11)

    plt.subplot(3,1,3)
    plt.plot(a[:,2], label="Arm")
    plt.plot(f[:,2], label="Forearm")
    plt.plot(h[:,2], label="Hand")
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel('Acceleration (m/s²)', fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Interpolated Accelerometer Data (Z axis)", fontsize=11)

    plt.subplots_adjust(hspace=0.5)


# PLOT THE RESULT ANGLES
for i in range(len(attempts)):
    plt.figure()
    plt.suptitle(f"{name.capitalize()}, attempt {attempts[i]}", fontsize=13, fontweight="bold")

    plt.plot(elbow_angles[i])
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel("Angle (degrees)", fontsize=9)
    plt.title("Elbow Angles", fontsize=11)

# ----------------------------------

for i in range(len(attempts)):
    plt.figure()
    plt.suptitle(f"{name.capitalize()}, attempt {attempts[i]}", fontsize=13, fontweight="bold")

    plt.plot(wrist_angles[i])
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel("Angle (degrees)", fontsize=9)
    plt.title("Wrist Angles", fontsize=11)


# PLOT THE VELOCITIES
for i in range(len(attempts)):
    plt.figure()
    plt.suptitle(f"{name.capitalize()}, attempt {attempts[i]}", fontsize=13, fontweight="bold")

    vel_x = [v[0] for v in arm_velocities[i]]
    vel_y = [v[1] for v in arm_velocities[i]]
    vel_z = [v[2] for v in arm_velocities[i]]

    plt.subplot(3,1,1)
    plt.plot(vel_x, label="X")
    plt.plot(vel_y, label="Y")
    plt.plot(vel_z, label="Z")
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel("Speed (m/s)", fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Arm Velocity", fontsize=11)

    vel_x = [v[0] for v in forearm_velocities[i]]
    vel_y = [v[1] for v in forearm_velocities[i]]
    vel_z = [v[2] for v in forearm_velocities[i]]

    plt.subplot(3,1,2)
    plt.plot(vel_x, label="X")
    plt.plot(vel_y, label="Y")
    plt.plot(vel_z, label="Z")
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel("Speed (m/s)", fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Forearm Velocity", fontsize=11)

    vel_x = [v[0] for v in hand_velocities[i]]
    vel_y = [v[1] for v in hand_velocities[i]]
    vel_z = [v[2] for v in hand_velocities[i]]

    plt.subplot(3,1,3)
    plt.plot(vel_x, label="X")
    plt.plot(vel_y, label="Y")
    plt.plot(vel_z, label="Z")
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel("Speed (m/s)", fontsize=9)
    plt.legend(loc="upper right", fontsize=8)
    plt.title("Hand Velocity", fontsize=11)

    plt.subplots_adjust(hspace=0.5)

# ----------------------------------

for i in range(len(attempts)):
    plt.figure()
    plt.suptitle(f"{name.capitalize()}, attempt {attempts[i]}", fontsize=13, fontweight="bold")

    plt.subplot(3,1,1)
    plt.plot(total_speed_arm[i])
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel("Speed (m/s)", fontsize=9)
    plt.title("Total Speed Arm", fontsize=11)

    plt.subplot(3,1,2)
    plt.plot(total_speed_forearm[i])
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel("Speed (m/s)", fontsize=9)
    plt.title("Total Speed Forearm", fontsize=11)

    plt.subplot(3,1,3)
    plt.plot(total_speed_hand[i])
    plt.xlabel('Time (samples)', fontsize=9)
    plt.ylabel("Speed (m/s)", fontsize=9)
    plt.title(f"User 6, attempt {attempts[i]}", fontsize=13, fontweight="bold")

    plt.subplots_adjust(hspace=0.5)


plt.show(block=False)
input("Press [enter] key to close plots...")