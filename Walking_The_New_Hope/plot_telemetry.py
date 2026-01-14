from matplotlib import pyplot as plt
plt.rcParams["text.usetex"] = False

import pandas as pd
from math import atan2, asin, copysign, pi
import os
from statistics import mean, stdev


class TelemetryPlotter:
  def __init__(self, filename):
    self.filename = filename
    self.data = None
    self.euler_roll = []
    self.euler_pitch = []
    self.euler_yaw = []
    self.timestamp = []

    self.load_data()
    self.get_timestamps()
    self.get_euler_orientation()

    # latex formatting
    plt.rcParams.update({
      "text.usetex": True,
      "font.family": "monospace",
      "font.monospace": 'Computer Modern Typewriter'
    })


  def load_data(self):
    if os.path.exists(self.filename) == False:
      print(f"File {self.filename} does not exist.")
      return

    self.data = pd.read_csv(self.filename)


  def cut_data(self, start_time, end_time):
    if self.data is None:
      print("No data loaded")
      return

    mask = (self.data['timestamp'] - self.data['timestamp'].iloc[0] >= start_time) & \
           (self.data['timestamp'] - self.data['timestamp'].iloc[0] <= end_time)
    self.data = self.data.loc[mask].reset_index(drop=True)
    self.get_timestamps()
    self.get_euler_orientation()

    print(f"Data cut to range {start_time} to {end_time}.")
    print(f"New data length: {len(self.data)} samples.")


  def get_timestamps(self):
    self.timestamp = self.data['timestamp'].to_list()
    self.timestamp = [(t - self.timestamp[0]) for t in self.timestamp] # Use relative time in seconds


  def get_euler_orientation(self):
    if self.data is None:
      print("No data loaded")
      return

    self.euler_roll = []
    self.euler_pitch = []
    self.euler_yaw = []

    for _, row in self.data.iterrows():
      qw = row['quat_w']
      qx = row['quat_x']
      qy = row['quat_y']
      qz = row['quat_z']

      # Roll (x-axis rotation)
      sinr_cosp = 2 * (qw * qx + qy * qz)
      cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
      roll = atan2(sinr_cosp, cosr_cosp)

      # Pitch (y-axis rotation)
      sinp = 2 * (qw * qy - qz * qx)
      if abs(sinp) >= 1:
        pitch = copysign(pi / 2, sinp)  # use 90 degrees if out of range
      else:
        pitch = asin(sinp)

      # Yaw (z-axis rotation)
      siny_cosp = 2 * (qw * qz + qx * qy)
      cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
      yaw = atan2(siny_cosp, cosy_cosp)

      self.euler_roll.append(roll * (180.0 / pi))
      self.euler_pitch.append(pitch * (180.0 / pi))
      self.euler_yaw.append(yaw * (180.0 / pi))


  def plot_orientation(self):
    if self.data is None:
      print("No data loaded")
      return

    self.get_euler_orientation()

    plt.figure(figsize=(10, 6))
    plt.plot(self.timestamp, self.euler_roll, label='Roll')
    plt.plot(self.timestamp, self.euler_pitch, label='Pitch')
    plt.plot(self.timestamp, self.euler_yaw, label='Yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('Orientation [$^\circ$]')
    plt.title('\\bf{Orientation data}')
    plt.legend()
    plt.grid()
    plt.show()


  def plot_gyro(self):
    if self.data is None:
      print("No data loaded")
      return

    self.gyro_x = self.data['gyro_x'].to_list()
    self.gyro_y = self.data['gyro_y'].to_list()
    self.gyro_z = self.data['gyro_z'].to_list()

    # Convert from rad/s to deg/s
    self.gyro_x = [g * (180.0 / pi) for g in self.gyro_x]
    self.gyro_y = [g * (180.0 / pi) for g in self.gyro_y]
    self.gyro_z = [g * (180.0 / pi) for g in self.gyro_z]

    plt.figure(figsize=(10, 6))
    plt.plot(self.timestamp, self.gyro_x, label='Gyro X')
    plt.plot(self.timestamp, self.gyro_y, label='Gyro Y')
    plt.plot(self.timestamp, self.gyro_z, label='Gyro Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [$\\frac{^\circ}{s}$]')
    plt.title('\\bf{Gyroscope data}')
    plt.legend()
    plt.grid()
    plt.show()


  def plot_accel(self):
    if self.data is None:
      print("No data loaded")
      return

    self.accel_x = self.data['accel_x'].to_list()
    self.accel_y = self.data['accel_y'].to_list()
    self.accel_z = self.data['accel_z'].to_list()

    plt.figure(figsize=(10, 6))
    plt.plot(self.timestamp, self.accel_x, label='Accel X')
    plt.plot(self.timestamp, self.accel_y, label='Accel Y')
    plt.plot(self.timestamp, self.accel_z, label='Accel Z')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [$\\frac{m}{s^2}$]')
    plt.title('\\bf{Accelerometer data}')
    plt.legend()
    plt.grid()
    plt.show()


  def get_statistics(self):
    if self.data is None:
      print("No data loaded")
      return

    stats = self.data.describe()
    print(stats)

    gyro_x_stats = self.data['gyro_x'].describe()
    print("Gyro X Statistics:")
    print(gyro_x_stats)

    gyro_y_stats = self.data['gyro_y'].describe()
    print("Gyro Y Statistics:")
    print(gyro_y_stats)

    gyro_z_stats = self.data['gyro_z'].describe()
    print("Gyro Z Statistics:")
    print(gyro_z_stats)

    roll_mean = mean(self.euler_roll)
    pitch_mean = mean(self.euler_pitch)
    yaw_mean = mean(self.euler_yaw)

    roll_stdev = stdev(self.euler_roll)
    pitch_stdev = stdev(self.euler_pitch)
    yaw_stdev = stdev(self.euler_yaw)

    print(f"Orientation Statistics:")
    print(f"Roll: Mean = {roll_mean:.2f} deg, Std Dev = {roll_stdev:.2f} deg")
    print(f"Pitch: Mean = {pitch_mean:.2f} deg, Std Dev = {pitch_stdev:.2f} deg")
    print(f"Yaw: Mean = {yaw_mean:.2f} deg, Std Dev = {yaw_stdev:.2f} deg")


def main():
  telem = TelemetryPlotter('log_01_walki_PID.csv')

  telem.cut_data(3.0, 12.0)

  telem.get_statistics()

  telem.plot_orientation()
  telem.plot_gyro()
  telem.plot_accel()


if __name__ == "__main__":
  main()
