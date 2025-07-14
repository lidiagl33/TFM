from mbientlab.metawear import MetaWear, libmetawear
from mbientlab.metawear.cbindings import *
from mbientlab.warble import *
from mbientlab.metawear import *
from time import sleep, time
import numpy as np
import sys
import keyboard
from calculate_data import calculate_data


# Create a state object to store information about the device and the number of data samples received
class State:
    def __init__(self, device):
        self.device = device
        self.sensor = ""
        self.samples_acc = 0
        self.samples_gyro = 0
        self.acc_callback = FnVoid_VoidP_DataP(self.data_handler_acc)
        self.gyro_callback = FnVoid_VoidP_DataP(self.data_handler_gyro)
        self.battery_callback = FnVoid_VoidP_DataP(self.battery_handler)
        self.acc_data = []
        self.gyro_data = []
        self.acc_data_filtered = []
        self.gyro_data_filtered = []
        self.acc_timestamps = []  # store the timestamps of the accelerometer
        self.gyro_timestamps = []  # store the timestamps of the gyroscope
    def reset_data(self):
        self.samples_acc = 0
        self.samples_gyro = 0
        self.acc_data = []
        self.gyro_data = []
        self.acc_data_filtered = []
        self.gyro_data_filtered = []
        self.acc_timestamps = []
        self.gyro_timestamps = []
    def data_handler_acc(self, ctx, data):
        values = cast(data.contents.value, POINTER(CartesianFloat)).contents # in g (terrestrial gravity)
        # Convert to m/s²
        x = values.x * 9.81
        y = values.y * 9.81
        z = values.z * 9.81
        timestamp = time() # get the current time
        # print("(" + self.sensor + ") " + f"Accelerometer [m/s²] -> x: {x}, y: {y}, z: {z}") # parse_value(data)
        self.samples_acc += 1
        self.acc_data.append([x, y, z])
        self.acc_timestamps.append(timestamp)
    def data_handler_gyro(self, ctx, data):
        values = cast(data.contents.value, POINTER(CartesianFloat)).contents # in dps (degrees per second)
        # Convert to rad/s
        x = values.x * np.pi/180
        y = values.y * np.pi/180
        z = values.z * np.pi/180
        timestamp = time() # get the current time
        # print("(" + self.sensor + ") " + f"Gyroscope [rad/s] -> x: {x}, y: {y}, z: {z}") # parse_value(data)
        self.samples_gyro += 1
        self.gyro_data.append([x, y, z])
        self.gyro_timestamps.append(timestamp)
    def battery_handler(self, ctx, data):
        print("Battery [mV, %]: ", parse_value(data))


def get_address(sensor):
    if sensor == "A":
        add = "E2:81:0C:F9:E3:0C"
        return add
    elif sensor == "B":
        add = "FA:79:5A:03:0C:03"
        return add
    elif sensor == "C":
        add = "DC:A3:78:95:CF:69" 
        return add
    elif sensor == "D":
        add = "C1:86:EB:CE:38:92"  
        return add
    elif sensor == "E":
        add = "C6:69:03:18:31:56"
        return add
    else:
        "Wrong sensor, try again"
        return None


def read_data(states):
    print("⏸️  Press 'ENTER' to start")
    keyboard.wait("enter")
    print("⏩ Reading data from the sensor... (Press 'SPACE' to stop)")
    start = time() # time of beginning
    for s in states:
        # Start streaming data
        libmetawear.mbl_mw_acc_enable_acceleration_sampling(s.device.board)
        libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(s.device.board)
        libmetawear.mbl_mw_acc_start(s.device.board) # switch accelerometer to active mode
        libmetawear.mbl_mw_gyro_bmi160_start(s.device.board) # switch gyroscope to active mode
    print("🔴 Receiving data...")
    while not keyboard.is_pressed("space"): # stop when "space" is pressed
        sleep(0.1) # little pause to avoid an excesive use of the CPU
    end = time() # time of ending
    reading_time = end - start # total time
    print(f"\n⏱️  END reading. Total time: {reading_time: .2f} s")


def reconnect_sensor(sensor, max_retries=5, delay=3):
    attempts = 0
    while attempts < max_retries:
        try:
            print(f"🔄 Trying to reconnect to sensor {sensor}... ({attempts+1}/{max_retries})")
            sensor.connect()
            print(f"✔️  Sensor {sensor.address} reconnected")
            return True
        except Exception as e:
            print(f"❌ Error reconnecting: {e}")
            sleep(delay)
            attempts += 1
    print(f"❌ Couldn't reconnect to sensor {sensor} after {max_retries} attempts")
    return False


def handle_disconnect(board):
    print(f"⚠️  Sensor {board.mac} has been disconnected.")
    # Here we can try to reconnect automatically
    for s in states:
        if s.device.board == board:
            if reconnect_sensor(s.device):  # if the reconnection is successful, keeps going
                break
            else:
                print(f"🔴 Fail reconnecting sensor {board.mac}")
                break


# python get_data.py 2 3 4
# 2 -> arm
# 3 -> forearm
# 4 -> hand

# 1️⃣ Connect the sensors (only 1 time)
states = [] # list of sensors
for i in range(len(sys.argv) - 1):
    BleScanner.set_handler(lambda result: print(result))
    BleScanner.start()
    sensor = sys.argv[i + 1]
    d = MetaWear(get_address(sensor))
    try:
        print("⏳ Trying to connect...")
        d.connect()
        # Add disconnection callback
        d.on_disconnect = lambda board: handle_disconnect(board) # call the function in case of disconnection (anytime during the execution)
        print("✔️  Connected to sensor " + sensor + " -> " + d.address)
        states.append(State(d))
        states[-1].sensor = sensor
        sleep(2)
    except WarbleException as e:
        print(f"❌ CONNECTION ERROR: failed to discover sensor {sensor}")
        sys.exit()

print("\n")

for s in states:
    print("Configuring device with address " + s.device.address)
    libmetawear.mbl_mw_settings_set_connection_parameters(s.device.board, 7.5, 7.5, 0, 6000)
    sleep(2)

    # Configure the accelerometer
    libmetawear.mbl_mw_acc_set_odr(s.device.board, 50.0) # set ouput data rate (50 Hz)
    libmetawear.mbl_mw_acc_set_range(s.device.board, 8.0) # set full scale range (8g), g's = acceleration in terrestrial gravity units -> 1g ≈ 9.81 m/s²
    libmetawear.mbl_mw_acc_write_acceleration_config(s.device.board)

    # Configure the gyroscope
    libmetawear.mbl_mw_gyro_bmi160_set_odr(s.device.board, 7) # set ouput data rate (50 Hz)
    libmetawear.mbl_mw_gyro_bmi160_set_range(s.device.board, 1) # set the rotation range (1000 dps = degrees per second)
    libmetawear.mbl_mw_gyro_bmi160_write_config(s.device.board)

    # Get the data signals and subscribe to the acc and gyro data streams
    acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board) # get accelerometer data
    gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(s.device.board) # get gyroscope data
    libmetawear.mbl_mw_datasignal_subscribe(acc_signal, None, s.acc_callback)
    libmetawear.mbl_mw_datasignal_subscribe(gyro_signal, None, s.gyro_callback)

# 2️⃣ Loop of shooting attempts
print("\n🎯 READY! Press 'ENTER' for start a new attempt or type 'EXIT' to stop")
attempt_counter = 31
name = ""
try:
    while True:
        print(f"\n🔍 Checking sensors' status...")
        # Verifying disconnections
        disconnected = False
        for s in states:
            s.reset_data()
            if not s.device.is_connected:
                print(f"❌ Sensor {s.sensor} disconnected.")
                disconnected = True
        if disconnected:
            print("⛔ Stopping the shooting register. Reconnect the sensors")
            break
        
        command = input()
        if command.strip().lower() == "exit":
            print("\n🔚 Ending the program...")
            print("🔌 Disconnecting the sensors...")
            # 3️⃣ Disconnect the sensors at the end
            for s in states:
                # Stop the acc and gyro
                libmetawear.mbl_mw_acc_stop(s.device.board)
                libmetawear.mbl_mw_gyro_bmi160_stop(s.device.board)

                # Disable acceleration and rotation sampling
                libmetawear.mbl_mw_acc_disable_acceleration_sampling(s.device.board)
                libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(s.device.board)

                acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
                gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(s.device.board)
                libmetawear.mbl_mw_datasignal_unsubscribe(acc_signal)
                libmetawear.mbl_mw_datasignal_unsubscribe(gyro_signal)

                libmetawear.mbl_mw_debug_disconnect(s.device.board)
            break

        while True:
            n = input("\nName of the person: ") # name of the shooter
            if n != "":
                if n != name:
                    name = n
                    attempt_counter = 31    
                else:
                    name = name
                    attempt_counter += 1
                break
            else:
                print("Invalid name. Try again")

        print(f"\n🟢 Attempt {attempt_counter}: starting recording...")            
        read_data(states)
        print(f"📐 Processing data of attempt #{attempt_counter}...")
        calculate_data(states,name,attempt_counter)
        print(f"✅ Attempt #{attempt_counter} processed")

except KeyboardInterrupt:
    print("⛔ Manual interruption. Disconnecting sensors...")
    # 3️⃣ Disconnect the sensors at the end
    for s in states:
        # Stop the acc and gyro
        libmetawear.mbl_mw_acc_stop(s.device.board)
        libmetawear.mbl_mw_gyro_bmi160_stop(s.device.board)

        # Disable acceleration and rotation sampling
        libmetawear.mbl_mw_acc_disable_acceleration_sampling(s.device.board)
        libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(s.device.board)

        acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
        gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(s.device.board)
        libmetawear.mbl_mw_datasignal_unsubscribe(acc_signal)
        libmetawear.mbl_mw_datasignal_unsubscribe(gyro_signal)

        libmetawear.mbl_mw_debug_disconnect(s.device.board)

print("\n\nEND\n")