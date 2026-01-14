import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from statistics import mean, stdev


class IMUTelemetryAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_csv(filename)

        # ensure sorted by time
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)

        # relative time
        self.data["time"] = self.data["timestamp"] - self.data["timestamp"].iloc[0]

    # -------------------------------------------------
    # Data cutting
    # -------------------------------------------------
    def cut_start(self, start_samples):
        """Cut the first start_samples samples from the dataset and rebase time."""
        start_samples = int(start_samples)
        if start_samples <= 0:
            print(f"No samples cut (requested {start_samples}). {len(self.data)} samples remain")
            return

        n = len(self.data)
        if start_samples >= n:
            # result is an empty dataframe with same columns
            self.data = self.data.iloc[0:0].reset_index(drop=True)
            print(f"Cut first {start_samples} samples → 0 samples remain")
            return

        self.data = self.data.iloc[start_samples:].reset_index(drop=True)
        if len(self.data) > 0:
            self.data["time"] = self.data["time"] - self.data["time"].iloc[0]
        print(f"Cut first {start_samples} samples → {len(self.data)} samples remain")

    def cut_end(self, end_samples):
        """Cut the last end_samples samples from the dataset and rebase time."""
        end_samples = int(end_samples)
        if end_samples <= 0:
            print(f"No samples cut (requested {end_samples}). {len(self.data)} samples remain")
            return

        n = len(self.data)
        if end_samples >= n:
            # result is an empty dataframe with same columns
            self.data = self.data.iloc[0:0].reset_index(drop=True)
            print(f"Cut last {end_samples} samples → 0 samples remain")
            return

        # keep everything except the last `end_samples` rows
        self.data = self.data.iloc[:-end_samples].reset_index(drop=True)
        if len(self.data) > 0:
            self.data["time"] = self.data["time"] - self.data["time"].iloc[0]
        print(f"Cut last {end_samples} samples → {len(self.data)} samples remain")

    # -------------------------------------------------
    # Pretty statistics
    # -------------------------------------------------
    def print_stats(self):
        print("\n=== BASIC STATISTICS ===")

        for name in ["gyro_x", "gyro_y", "gyro_z"]:
            vals = self.data[name]
            print(
                f"{name:8s} | mean={mean(vals): .4f} rad/s "
                f"std={stdev(vals): .4f} "
                f"min={min(vals): .4f} "
                f"max={max(vals): .4f}"
            )

        for name in ["accel_x", "accel_y", "accel_z"]:
            vals = self.data[name]
            print(
                f"{name:8s} | mean={mean(vals): .4f} m/s² "
                f"std={stdev(vals): .4f} "
                f"min={min(vals): .4f} "
                f"max={max(vals): .4f}"
            )

        # time statistics
        t = self.data["time"]
        n = len(t)
        duration = float(t.iloc[-1]) if n > 0 else 0.0
        if n > 1:
            dts = np.diff(t.to_numpy())
            mean_dt = float(np.mean(dts))
            min_dt = float(np.min(dts))
            max_dt = float(np.max(dts))
            fs = 1.0 / mean_dt if mean_dt > 0 else float("nan")
        else:
            mean_dt = min_dt = max_dt = float("nan")
            fs = float("nan")

        print(
            f"time     | samples={n} duration={duration:.4f} s "
            f"mean_dt={mean_dt:.6f} s min_dt={min_dt:.6f} s max_dt={max_dt:.6f} s fs~={fs:.2f} Hz"
        )

    # -------------------------------------------------
    # Time-domain plots
    # -------------------------------------------------
    def plot_time(self):
        t = self.data["time"]

        plt.figure(figsize=(10, 6))
        plt.plot(t.to_numpy(), self.data["gyro_x"].to_numpy(), label="Gyro X")
        plt.plot(t.to_numpy(), self.data["gyro_y"].to_numpy(), label="Gyro Y")
        plt.plot(t.to_numpy(), self.data["gyro_z"].to_numpy(), label="Gyro Z")
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity [rad/s]")
        plt.title("Gyroscope – Time domain")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(t.to_numpy(), self.data["accel_x"].to_numpy(), label="Accel X")
        plt.plot(t.to_numpy(), self.data["accel_y"].to_numpy(), label="Accel Y")
        plt.plot(t.to_numpy(), self.data["accel_z"].to_numpy(), label="Accel Z")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [m/s²]")
        plt.title("Accelerometer – Time domain")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------
    # FFT utility
    # -------------------------------------------------
    def _fft(self, signal, fs):
        n = len(signal)
        signal = signal - np.mean(signal)  # remove DC
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(fft_vals) / n
        return freqs, mag

    # -------------------------------------------------
    # FFT plots
    # -------------------------------------------------
    def plot_fft(self):
        t = self.data["time"].values
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt

        print(f"\nEstimated sampling frequency: {fs:.1f} Hz")

        plt.figure(figsize=(20, 8))
        for axis in ["gyro_x", "gyro_y", "gyro_z"]:
            f, mag = self._fft(self.data[axis].values, fs)
            plt.plot(f, mag, label=axis)

        plt.xlim(0, 20)  # walking dynamics range
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.title("Gyroscope FFT")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20, 8))
        for axis in ["accel_x", "accel_y", "accel_z"]:
            f, mag = self._fft(self.data[axis].values, fs)
            plt.plot(f, mag, label=axis)

        plt.xlim(0, 20)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.title("Accelerometer FFT")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    analyzer = IMUTelemetryAnalyzer("telemetry_jan.csv")

    analyzer.cut_start(3000)   # cut first 3 seconds
    analyzer.cut_end(6000)
    analyzer.print_stats()
    analyzer.plot_time()
    analyzer.plot_fft()
