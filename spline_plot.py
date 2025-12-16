import time
import sys
import argparse
import numpy as np
from scipy.interpolate import CubicSpline


class SplineTrajectoryGenerator:
    """Generates cubic splines for swing lift and forward/back foot placement"""
    def __init__(self, lift_amplitude=0.12, forward_amplitude=0.06):
        # Reduced amplitudes and more conservative defaults
        self.lift_amplitude = lift_amplitude
        self.forward_amplitude = forward_amplitude

    def generate_swing_splines(self):
        # time knots (0..1)
        t = np.array([0.0, 0.3, 0.5, 0.7, 1.0])


        # lift: same shape (lift in middle)
        lift_points = np.array([0.0, 1.9, 1.0, 0.0, 0.0])

        # FORWARD: symmetric, less extreme -> avoids introducing net bias
        # starts slightly back, passes under body, ends slightly forward
        forward_points = np.array([-0.8, -0.3, 0.8, 0.8, -0.8])

        lift_spline = CubicSpline(t, lift_points, bc_type='clamped')
        forward_spline = CubicSpline(t, forward_points, bc_type='clamped')
        return lift_spline, forward_spline

import matplotlib.pyplot as plt


def main():
    gen = SplineTrajectoryGenerator()  # uses lift_amplitude and forward_amplitude
    lift_spline, forward_spline = gen.generate_swing_splines()

    t = np.linspace(0.0, 1.0, 400)
    lift = lift_spline(t) * gen.lift_amplitude
    forward = forward_spline(t) * gen.forward_amplitude

    # plot time series
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
    axs[0].plot(t, lift, label="lift (m)")
    axs[0].set_ylabel("lift (m)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, forward, label="forward (m)", color="tab:orange")
    axs[1].set_xlabel("normalized time")
    axs[1].set_ylabel("forward (m)")
    axs[1].grid(True)
    axs[1].legend()

    # optional: show 2D foot path (forward vs lift)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(forward, lift, "-k")
    # plot knot positions (scaled)
    knot_t = np.array([0.0, 0.3, 0.7, 1.0])
    # access raw knot values by evaluating splines at knot_t
    ax2.scatter(forward_spline(knot_t) * gen.forward_amplitude,
                lift_spline(knot_t) * gen.lift_amplitude,
                c="red", zorder=5, label="knots")
    ax2.set_xlabel("forward (m)")
    ax2.set_ylabel("lift (m)")
    ax2.set_title("Swing foot path (forward vs lift)")
    ax2.grid(True)
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    main()