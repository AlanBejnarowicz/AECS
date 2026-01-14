import csv
from datetime import datetime


class UnitreeLowStateIMULogger:
    def __init__(self, filename=None):
        self.filename = filename or self._default_filename()
        self._open_file()

    def _default_filename(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"telemetry_log_{ts}.csv"

    def _open_file(self):
        self.csv_file = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            "timestamp",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z",
        ])

    def log(self, lowstate, sim_time):
        """
        Call once per control step.
        lowstate : LowState_
        sim_time : float (seconds, e.g. data.time)
        """

        imu = lowstate.imu_state

        # Unitree IMU convention:
        # quaternion: [w, x, y, z]
        qw = imu.quaternion[0]
        qx = imu.quaternion[1]
        qy = imu.quaternion[2]
        qz = imu.quaternion[3]

        # gyro: rad/s
        gx = imu.gyroscope[0]
        gy = imu.gyroscope[1]
        gz = imu.gyroscope[2]

        # accel: m/s^2
        ax = imu.accelerometer[0]
        ay = imu.accelerometer[1]
        az = imu.accelerometer[2]

        self.writer.writerow([
            sim_time,
            qw, qx, qy, qz,
            gx, gy, gz,
            ax, ay, az,
        ])

    def close(self):
        self.csv_file.close()
