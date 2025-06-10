import csv
import time
from datetime import datetime
from mbientlab.metawear import MetaWear, libmetawear
from mbientlab.metawear.cbindings import *


# Parameters
mac_addresses = ["C5:AE:F0:9E:3D:37"]
tx_power = -20 # [dBm] -> [-20, 0, 4]
filename = "changes_battery_"+str(tx_power)+"dBm.csv"

class SensorWrapper:
    def __init__(self, mac, name):
        self.mac = mac
        self.name = name
        self.device = MetaWear(mac)
        self.connected = False
        self.streaming = False
        self.battery_callback = FnVoid_VoidP_DataP(self.battery_handler)
        self.acc_callback = FnVoid_VoidP_DataP(self.acc_handler)
        self.gyro_callback = FnVoid_VoidP_DataP(self.gyro_handler)
        self.data = []

    def connect(self):
        self.device.connect()
        self.connected = True
        libmetawear.mbl_mw_settings_set_tx_power(self.device.board, tx_power)
        libmetawear.mbl_mw_settings_set_connection_parameters(self.device.board, 7.5, 7.5, 0, 6000)
        time.sleep(1)
        print(f"✔️  {self.name} connected with {tx_power} dBm")

    def begin_streaming(self):
        # Accelerometer
        libmetawear.mbl_mw_acc_set_odr(self.device.board, 50.0)
        libmetawear.mbl_mw_acc_set_range(self.device.board, 8.0)
        libmetawear.mbl_mw_acc_write_acceleration_config(self.device.board)
        acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.device.board)
        libmetawear.mbl_mw_datasignal_subscribe(acc_signal, None, self.acc_callback)
        libmetawear.mbl_mw_acc_enable_acceleration_sampling(self.device.board)
        libmetawear.mbl_mw_acc_start(self.device.board)
        # Gyroscope
        libmetawear.mbl_mw_gyro_bmi160_set_odr(self.device.board, 7)
        libmetawear.mbl_mw_gyro_bmi160_set_range(self.device.board, 1)
        libmetawear.mbl_mw_gyro_bmi160_write_config(self.device.board)
        gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(self.device.board)
        libmetawear.mbl_mw_datasignal_subscribe(gyro_signal, None, self.gyro_callback)
        libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(self.device.board)
        libmetawear.mbl_mw_gyro_bmi160_start(self.device.board)

        self.streaming = True
        print(f"▶️  {self.name} began to transmit")

    def get_battery(self):
        if self.connected:
            battery_signal = libmetawear.mbl_mw_settings_get_battery_state_data_signal(self.device.board)
            libmetawear.mbl_mw_datasignal_subscribe(battery_signal, None, self.battery_callback)
            libmetawear.mbl_mw_datasignal_read(battery_signal)

    def battery_handler(self, ctx, data):
        data_ptr = cast(data, POINTER(Data)).contents
        battery_state = cast(data_ptr.value, POINTER(BatteryState)).contents
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.data.append((self.name, tx_power, timestamp, battery_state.charge, battery_state.voltage))
        print(f"🔋 {self.name} - {battery_state.charge}% / {battery_state.voltage} mV")

    def acc_handler(self, ctx, data):
        ""
    
    def gyro_handler(self, ctx, data):
        ""

    def disconnect(self):
        if self.streaming:
            # Accelerometer
            libmetawear.mbl_mw_acc_stop(self.device.board)
            libmetawear.mbl_mw_acc_disable_acceleration_sampling(self.device.board)
            acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(self.device.board)
            libmetawear.mbl_mw_datasignal_unsubscribe(acc_signal)
            # Gyroscope
            libmetawear.mbl_mw_gyro_bmi160_stop(self.device.board)
            libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(self.device.board)
            gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(self.device.board)
            libmetawear.mbl_mw_datasignal_unsubscribe(gyro_signal)

            libmetawear.mbl_mw_debug_disconnect(self.device.board)

        self.connected = False
        print(f"❌ {self.name} disconnected")


def save_data(sensor, writer, f):
    if len(sensor.data) > 0:
        for row in sensor.data:
            writer.writerow(row)
            f.flush()
        print(f"💾 {sensor.name}: {len(sensor.data)} rows saved")
    else:
        print(f"⚠️ {sensor.name}: no data to save")


def stop_streaming(sensor):
    libmetawear.mbl_mw_datasignal_unsubscribe(libmetawear.mbl_mw_settings_get_battery_state_data_signal(sensor.device.board))


# Initialize sensors
sensors = [SensorWrapper(mac, f"Sensor {i+1}") for i, mac in enumerate(mac_addresses)]


# CSV: create a file
with open(filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Sensor", "Power_dBm", "Time", "Battery_%", "Voltage_mV"])
    f.flush()

    try:
        # Connect all
        for sensor in sensors:
            sensor.connect()
            sensor.begin_streaming()

        print("\n⏱️ Initializing periodical measurements of the battery...\n")

        try:
            while any(s.connected for s in sensors):
                for sensor in sensors:
                    if sensor.connected:
                        try:
                            sensor.get_battery()
                            time.sleep(2)
                        except Exception as e:
                            print(f"⚠️ Error measuring the battery in {sensor.name}: {e}")
                time.sleep(300) # wait 5 min
                
        except KeyboardInterrupt:
            print("\n🛑 Manual interruption detected (Ctrl+C). Ending experiment...")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    finally:
        # Disconnect all and save the CSV
        for sensor in sensors:
            try:
                print(f"📝 Saving data of {sensor.name}")
                save_data(sensor, writer, f)
                print(f"🔌 Disconnecting {sensor.name}...")
                stop_streaming(sensor)
            except Exception as e:
                print(f"❌ Error when saving data of {sensor.name}: {e}")

            try:                    
                sensor.disconnect()
            except Exception as e:
                print(f"⚠️ Error when disconnecting {sensor.name}: {e}")
                    

print(f"\n✅ Data saved in '{filename}'")