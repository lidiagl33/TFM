from time import sleep
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


def butter_lowpass_filter(data, cutoff=3, fs=50, order=5):
    nyq = 0.5*fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data) # same number of samples (doesn't change the size of the signal)


def complementary_filter(previous_angle, w, a, alpha, dt):
    # Filter the orientation combining the gyroscope and the accelerometer
    gyro_angle = previous_angle + w * dt # rad
    acc_angle = a # rad

    final_angle = alpha * gyro_angle + (1-alpha)*acc_angle # rad

    return final_angle


def get_angle_between_segments(v1, v2):
    # Calculate the angle between two 3D vectors using the dot product
    dot_product = np.dot(v1,v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_theta = np.clip(dot_product/(norm_v1*norm_v2), -1.0, 1.0) # avoid numeric errors
    angle = np.arccos(cos_theta)*(180/np.pi) # deg

    return angle


def moving_average(data, window_size=9):
    # Make the filtered signal smaller (less samples)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def calculate_data(states, name, attempt_counter):
    print("\n📊 Total Samples Received:")
    for s in states:
        print("Accelerometer of %s (sensor %s) -> %d" % (s.device.address, s.sensor, s.samples_acc))
        print("Gyroscope of %s (sensor %s) -> %d" % (s.device.address, s.sensor, s.samples_gyro))

    sleep(1)
    print("\n------------------\n")

    # Filter the accelerometer and gyroscope data
    for s in states:
        # Convert to arrays
        acc_x = [d[0] for d in s.acc_data]
        acc_y = [d[1] for d in s.acc_data]
        acc_z = [d[2] for d in s.acc_data]

        gyro_x = [d[0] for d in s.gyro_data]
        gyro_y = [d[1] for d in s.gyro_data]
        gyro_z = [d[2] for d in s.gyro_data]

        # Apply Butterworth filter
        acc_x_filt = butter_lowpass_filter(acc_x)
        acc_y_filt = butter_lowpass_filter(acc_y)
        acc_z_filt = butter_lowpass_filter(acc_z)

        gyro_x_filt = butter_lowpass_filter(gyro_x)
        gyro_y_filt = butter_lowpass_filter(gyro_y)
        gyro_z_filt = butter_lowpass_filter(gyro_z)

        s.acc_data_filtered = list(zip(acc_x_filt, acc_y_filt, acc_z_filt))
        s.gyro_data_filtered = list(zip(gyro_x_filt, gyro_y_filt, gyro_z_filt))
        
    w_arm, w_forearm, w_hand = [], [], []
    a_arm, a_forearm, a_hand = [], [], []
    t_arm_gyro, t_forearm_gyro, t_hand_gyro = [], [], [] # timestamps gyro
    t_arm_acc, t_forearm_acc, t_hand_acc = [], [], [] # timestamps acc

    # Extract data with timestamps
    for i in range(len(states[0].gyro_data_filtered)):
        # Angular velocity (w) -> 3 axis
        w_arm.append(states[0].gyro_data_filtered[i])  # rad/s
        t_arm_gyro.append(states[0].gyro_timestamps[i])

    for i in range(len(states[1].gyro_data_filtered)):
        # Angular velocity (w) -> 3 axis
        w_forearm.append(states[1].gyro_data_filtered[i]) # rad/s
        t_forearm_gyro.append(states[1].gyro_timestamps[i])

    for i in range(len(states[2].gyro_data_filtered)):
        # Angular velocity (w) -> 3 axis
        w_hand.append(states[2].gyro_data_filtered[i]) # rad/s
        t_hand_gyro.append(states[2].gyro_timestamps[i])

    for i in range(len(states[0].acc_data_filtered)):
        # Linear acceleration (a) -> 3 axis
        a_arm.append(states[0].acc_data_filtered[i])  # m/s²
        t_arm_acc.append(states[0].acc_timestamps[i])

    for i in range(len(states[1].acc_data_filtered)):
        # Linear acceleration (a) -> 3 axis
        a_forearm.append(states[1].acc_data_filtered[i]) # m/s²
        t_forearm_acc.append(states[1].acc_timestamps[i])

    for i in range(len(states[2].acc_data_filtered)):
        # Linear acceleration (a) -> 3 axis
        a_hand.append(states[2].acc_data_filtered[i]) # m/s²
        t_hand_acc.append(states[2].acc_timestamps[i])

    # Save filtered data
    np.save('./plot/filtered/a_arm_' + name + '_' + str(attempt_counter) + '.npy', a_arm)
    np.save('./plot/filtered/w_arm_' + name + '_' + str(attempt_counter) + '.npy', w_arm)
    np.save('./plot/filtered/a_forearm_' + name + '_' + str(attempt_counter) + '.npy', a_forearm)
    np.save('./plot/filtered/w_forearm_' + name + '_' + str(attempt_counter) + '.npy', w_forearm)
    np.save('./plot/filtered/a_hand_' + name + '_' + str(attempt_counter) + '.npy', a_hand)
    np.save('./plot/filtered/w_hand_' + name + '_' + str(attempt_counter) + '.npy', w_hand)


    # Calculate difference between consecutive timestamps (all equal)
    dt_arm_gyro = np.diff(t_arm_gyro)
    # dt_forearm_gyro = np.diff(t_forearm_gyro)
    # dt_arm_acc = np.diff(t_arm_acc)
    # dt_forearm_acc = np.diff(t_forearm_acc)

    # Show statistics of the previous calculated dt
    # print(f"average dt_arm_gyro: {np.round(np.mean(dt_arm_gyro),2)}, std: {np.std(dt_arm_gyro)}")

    # Create a common time scale (based on the total range of timestamps)
    t_min = max(min(t_arm_gyro), min(t_forearm_gyro), min(t_hand_gyro), min(t_arm_acc), min(t_forearm_acc), min(t_hand_acc))
    t_max = min(max(t_arm_gyro), max(t_forearm_gyro), max(t_hand_gyro), max(t_arm_acc), max(t_forearm_acc), max(t_hand_acc))
    num_samples = min(len(w_arm), len(w_forearm), len(w_hand), len(a_arm), len(a_forearm), len(a_hand))  # based on the sensor with less data samples
    
    t_common = np.linspace(t_min, t_max, num_samples)  # new uniform time scale

    # Interpolate all the signals with the common time scale
    interp_w_arm = interp1d(t_arm_gyro, np.array(w_arm), axis=0, kind='linear', fill_value="extrapolate")
    interp_w_forearm = interp1d(t_forearm_gyro, np.array(w_forearm), axis=0, kind='linear', fill_value="extrapolate")
    interp_w_hand = interp1d(t_hand_gyro, np.array(w_hand), axis=0, kind='linear', fill_value="extrapolate")
    interp_a_arm = interp1d(t_arm_acc, np.array(a_arm), axis=0, kind='linear', fill_value="extrapolate")
    interp_a_forearm = interp1d(t_forearm_acc, np.array(a_forearm), axis=0, kind='linear', fill_value="extrapolate")
    interp_a_hand = interp1d(t_hand_acc, np.array(a_hand), axis=0, kind='linear', fill_value="extrapolate")

    # Evaluate the interpolated functions in the common time scale
    w_arm_int = interp_w_arm(t_common)
    w_forearm_int = interp_w_forearm(t_common)
    w_hand_int = interp_w_hand(t_common)
    a_arm_int = interp_a_arm(t_common)
    a_forearm_int = interp_a_forearm(t_common)
    a_hand_int = interp_a_hand(t_common)

    # Save interpolated data
    np.save('./plot/interpolated/a_arm_sync_' + name + '_' + str(attempt_counter) + '.npy', a_arm_int)
    np.save('./plot/interpolated/w_arm_sync_' + name + '_' + str(attempt_counter) + '.npy', w_arm_int)
    np.save('./plot/interpolated/a_forearm_sync_' + name + '_' + str(attempt_counter) + '.npy', a_forearm_int)
    np.save('./plot/interpolated/w_forearm_sync_' + name + '_' + str(attempt_counter) + '.npy', w_forearm_int)
    np.save('./plot/interpolated/a_hand_sync_' + name + '_' + str(attempt_counter) + '.npy', a_hand_int)
    np.save('./plot/interpolated/w_hand_sync_' + name + '_' + str(attempt_counter) + '.npy', w_hand_int)

    # Give the index of the axis (0=X, 1=Y, 2=Z) with the highest angular velocity at each time
    dominant_axis_arm, dominant_axis_forearm, dominant_axis_hand = [], [], []
    for i in range(len(w_arm_int)):
        dominant_axis_arm.append(np.argmax(np.abs(w_arm_int[i]), axis=0))
        dominant_axis_forearm.append(np.argmax(np.abs(w_forearm_int[i]), axis=0))
        dominant_axis_hand.append(np.argmax(np.abs(w_hand_int[i]), axis=0))

    # Obtain the angular velocity in the dominant axis at each time instant
    w_arm_dominant = w_arm_int[np.arange(len(w_arm_int)), dominant_axis_arm]
    w_forearm_dominant = w_forearm_int[np.arange(len(w_forearm_int)), dominant_axis_forearm]
    w_hand_dominant = w_hand_int[np.arange(len(w_hand_int)), dominant_axis_hand]

    samples = len(w_arm_int)
    dt = np.round(np.mean(dt_arm_gyro),2) # time interval -> dt = 1/fs = 1/50 = 0.02 s

    # Thresholds to detect "no movement"
    gyro_threshold = 0.025  # rad/s
    acc_threshold = 0.7  # m/s²

    # Number of initial samples to estimate the gravity
    N = 10

    # DYNAMIC ESTIMATION OF THE GRAVITY (lowpass filter)
    alpha_g = 0.95 # filtering factor for gravity (common 0.95 to 0.99)
    # Initialize the estimation of the gravity with the first N samples
    gravity_arm_est = np.mean([states[0].acc_data_filtered[i] for i in range(N)], axis=0)
    gravity_forearm_est = np.mean([states[1].acc_data_filtered[i] for i in range(N)], axis=0)
    gravity_hand_est = np.mean([states[2].acc_data_filtered[i] for i in range(N)], axis=0)

    arm_angle, forearm_angle, hand_angle = 0, 0, 0 # initial angle (rad)
    angles_arm, angles_forearm, angles_hand = [], [], []
    v_arm, v_forearm, v_hand = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] # initial lineal velocities (x,y,z)
    velocities_arm, velocities_forearm, velocities_hand = [], [], []
    total_speed_arm, total_speed_forearm, total_speed_hand = [], [], []

    for i in range(samples):
        # 1) Update the estimation of the gravity with a lowpass filter
        # During the first N values, it is assumed that the measured acceleration is the gravity 
        # (avoid errors if the sensor were moving at the beginning)
        if i < N:
            gravity_arm_est = a_arm_int[i]
            gravity_forearm_est = a_forearm_int[i]
            gravity_hand_est = a_hand_int[i]
        else:
            # Apply lowpass filter
            gravity_arm_est = alpha_g * gravity_arm_est + (1-alpha_g) * a_arm_int[i]
            gravity_forearm_est = alpha_g * gravity_forearm_est + (1-alpha_g) * a_forearm_int[i]
            gravity_hand_est = alpha_g * gravity_hand_est + (1-alpha_g) * a_hand_int[i]

        # 2) Dynamic acceleration = total_acceleration - estimated_gravity
        # (if the sensor is resting it should be close to 0 since the dynamic part is nul)
        a_arm_corrected = a_arm_int[i] - gravity_arm_est 
        a_forearm_corrected = a_forearm_int[i] - gravity_forearm_est
        a_hand_corrected = a_hand_int[i] - gravity_hand_est

        # 3) Detect resting (threshold of dynamic acceleration and gyro)
        resting_arm = (np.linalg.norm(a_arm_corrected) < acc_threshold and 
                    np.abs(w_arm_dominant[i]) < gyro_threshold)        
        resting_forearm = (np.linalg.norm(a_forearm_corrected) < acc_threshold and 
                        np.abs(w_forearm_dominant[i]) < gyro_threshold)        
        resting_hand = (np.linalg.norm(a_hand_corrected) < acc_threshold and 
                        np.abs(w_hand_dominant[i]) < gyro_threshold)

        # Check if the acceleration and gyroscope values are below the thresholds
        if resting_arm and resting_forearm and resting_hand:
            # If the sensor is resting, the angles are restored from the estimation of gravity
            new_arm_angle = np.arctan2(gravity_arm_est[2], gravity_arm_est[0]) # rad
            new_forearm_angle = np.arctan2(gravity_forearm_est[2], gravity_forearm_est[0]) # rad
            new_hand_angle = np.arctan2(gravity_hand_est[2], gravity_hand_est[0]) # rad

            # Smooth the transition instead of abruptly restoring
            arm_angle = 0.85*arm_angle + 0.15*new_arm_angle # rad
            forearm_angle = 0.85*forearm_angle + 0.15*new_forearm_angle # rad
            hand_angle = 0.85*hand_angle + 0.15*new_hand_angle # rad

        else:
            # 4) Calculate the angle of the accelerometer (XZ plane) using the estimated gravity
            arm_angle_acc = np.arctan2(gravity_arm_est[2], gravity_arm_est[0]) # rad
            forearm_angle_acc = np.arctan2(gravity_forearm_est[2], gravity_forearm_est[0]) # rad
            hand_angle_acc = np.arctan2(gravity_hand_est[2], gravity_hand_est[0])

            # 5) Dynamic adjustment of alpha for the complementary filter
            g = 9.81
            # Obtain dynamic acceleration magnitude
            a_total_arm = np.linalg.norm(a_arm_corrected)
            a_total_forearm = np.linalg.norm(a_forearm_corrected)
            a_total_hand = np.linalg.norm(a_hand_corrected)

            # Define an "error factor regarding to the gravity" (from 0 (close to g) to higher values)
            error_g_arm = a_total_arm / g
            error_g_forearm = a_total_forearm / g
            error_g_hand = a_total_hand / g

            # Define alpha as:
            # - small when error_g is 0
            # - big when error_g es high
            alpha_min, alpha_max = 0.20, 0.98
            alpha_arm = alpha_min + (alpha_max-alpha_min)*error_g_arm
            alpha_forearm = alpha_min + (alpha_max-alpha_min)*error_g_forearm
            alpha_hand = alpha_min + (alpha_max-alpha_min)*error_g_hand

            # Avoid exceeding the range [alpha_min, alpha_max]
            alpha_arm = np.clip(alpha_arm, alpha_min, alpha_max)
            alpha_forearm = np.clip(alpha_forearm, alpha_min, alpha_max)
            alpha_hand = np.clip(alpha_hand, alpha_min, alpha_max)

            # 6) Apply the complementary filter with the angular velocity in the dominant axis
            arm_angle = complementary_filter(arm_angle, w_arm_dominant[i], arm_angle_acc, alpha_arm, dt=dt) # rad
            forearm_angle = complementary_filter(forearm_angle, w_forearm_dominant[i], forearm_angle_acc, alpha_forearm, dt=dt) # rad
            hand_angle = complementary_filter(hand_angle, w_hand_dominant[i], hand_angle_acc, alpha_hand, dt=dt) # rad

        angles_arm.append(arm_angle) # rad
        angles_forearm.append(forearm_angle) # rad
        angles_hand.append(hand_angle) # rad

        # *) Obtain the lineal velocity: v[n] = v[n-1] + a_dyn[n]*dt
        v_arm += a_arm_corrected * dt
        v_forearm += a_forearm_corrected * dt
        v_hand += a_hand_corrected * dt

        # **) Drift correction if resting is detected (new thresholds) -> ZUPT (Zero Velocity Update)
        resting_arm_v = (np.linalg.norm(a_arm_corrected) < 0.3 and
            np.abs(w_arm_dominant[i]) < 0.01)
        resting_forearm_v = (np.linalg.norm(a_forearm_corrected) < 0.3 and
            np.abs(w_forearm_dominant[i]) < 0.01)
        resting_hand_v = (np.linalg.norm(a_hand_corrected) < 0.3 and
            np.abs(w_hand_dominant[i]) < 0.01)

        if resting_arm_v: # ZUPT activated
            v_arm = [0.0, 0.0, 0.0]
        if resting_forearm_v:
            v_forearm = [0.0, 0.0, 0.0]
        if resting_hand_v:
            v_hand = [0.0, 0.0, 0.0]

        # Current velocities
        velocities_arm.append(v_arm)
        velocities_forearm.append(v_forearm)
        velocities_hand.append(v_hand)

        speed_arm = np.linalg.norm(v_arm)
        speed_forearm = np.linalg.norm(v_forearm)
        speed_hand = np.linalg.norm(v_hand)

        total_speed_arm.append(speed_arm)
        total_speed_forearm.append(speed_forearm)
        total_speed_hand.append(speed_hand)

    # Convert angles in unitary vectors -> (cos, sin, 0)
    vectors_arm, vectors_forearm, vectors_hand = [], [], []
    for angle_arm, angle_forearm, angle_hand in zip(angles_arm, angles_forearm, angles_hand):
        vectors_arm.append(np.array([np.cos(angle_arm), np.sin(angle_arm), 0]))
        vectors_forearm.append(np.array([np.cos(angle_forearm), np.sin(angle_forearm), 0]))
        vectors_hand.append(np.array([np.cos(angle_hand), np.sin(angle_hand), 0]))

    # ELBOW ANGLE
    angles_elbow = [] # deg
    for v1, v2 in zip(vectors_arm, vectors_forearm):
        angles_elbow.append(get_angle_between_segments(v1, v2)) # deg

    # WRIST ANGLE
    angles_wrist = [] # deg
    for v1, v2 in zip(vectors_forearm, vectors_hand):
        angles_wrist.append(get_angle_between_segments(v1, v2)) # deg

    # Filter the data of the resulting angles
    smoothed_wrist_angle = moving_average(angles_wrist)
    smoothed_elbow_angle = moving_average(angles_elbow)

    # Save resulting data
    np.save('./plot/results/elb_ang_' + name + '_' + str(attempt_counter) + '.npy', smoothed_elbow_angle)
    np.save('./plot/results/wri_ang_' + name + '_' + str(attempt_counter) + '.npy', smoothed_wrist_angle)
    np.save('./plot/results/vel_arm_' + name + '_' + str(attempt_counter) + '.npy', velocities_arm)
    np.save('./plot/results/vel_forearm_' + name + '_' + str(attempt_counter) + '.npy', velocities_forearm)
    np.save('./plot/results/vel_hand_' + name + '_' + str(attempt_counter) + '.npy', velocities_hand)
    np.save('./plot/results/tot_vel_arm_' + name + '_' + str(attempt_counter) + '.npy', total_speed_arm)
    np.save('./plot/results/tot_vel_forearm_' + name + '_' + str(attempt_counter) + '.npy', total_speed_forearm)
    np.save('./plot/results/tot_vel_hand_' + name + '_' + str(attempt_counter) + '.npy', total_speed_hand)

    # Save main data
    np.save('./data/elbow_angles/elb_ang_' + name + '_' + str(attempt_counter) + '.npy', smoothed_elbow_angle)
    np.save('./data/wrist_angles/wri_ang_' + name + '_' +  str(attempt_counter) + '.npy', smoothed_wrist_angle)
    np.save('./data/wrist_velocities/wri_vel_' + name + '_' +  str(attempt_counter) + '.npy', total_speed_hand)

    input("")
    label = ''
    while True:
        label = input("Write the result of the free throw [in/out]: ")
        if label in ["in", "out"]:
            print("\nOK\n")
            break
        else:
            print("Invalid input. Please enter 'in' or 'out'.\n") # shooting result (score/miss)

    if label == 'in':
        np.save('./data/classes/class_' + name + '_' +  str(attempt_counter) + '.npy', 1)
    else:
        np.save('./data/classes/class_' + name + '_' +  str(attempt_counter) + '.npy', 0)
