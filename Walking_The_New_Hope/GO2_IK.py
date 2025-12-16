#!/usr/bin/env python3
"""
Go2 Leg IK class

Provides a clean, well-documented Python class `Go2LegIK` that encapsulates
full 3-DOF analytic inverse kinematics for a Unitree Go2-style leg and a
constrained "vertical-only" IK variant that preserves forward x to avoid
fore/aft base motion.

Usage example at the bottom shows how to create an instance and call both
`full_leg_ik` and `vertical_only_ik`.

Conventions:
 - Input coordinates (x, y, z) are in the HIP frame: x forward, y left, z down (meters).
 - Returned angles are (hip_abd, hip_pitch, knee_internal) in radians. Apply
   `calf_sign` if your motor expects an inverted sign for the calf joint.

IMPORTANT: tune `l_thigh` and `l_calf` to measured link lengths before running
on hardware. Test with the robot supported/harnessed.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class IKResult:
    hip_abd: float
    hip_pitch: float
    knee: float
    reachable: bool


class Go2LegIK:
    """Encapsulate Go2 leg inverse kinematics.

    Parameters
    ----------
    l_thigh : float
        Hip->knee length in meters.
    l_calf : float
        Knee->foot length in meters.
    hip_abd_sign : float
        Sign multiplier for hip abduction output (1 or -1) depending on joint convention.
    calf_sign : float
        Sign multiplier for calf (knee) output to match motor convention.
    """

    EPS = 1e-9

    def __init__(self, l_thigh=0.213, l_calf=0.213, hip_abd_sign=1.0, calf_sign=-1.0):
        self.l_thigh = float(l_thigh)
        self.l_calf = float(l_calf)
        self.hip_abd_sign = float(hip_abd_sign)
        self.calf_sign = float(calf_sign)

    # ----------------------- utility helpers -----------------------
    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def rot_x_vec(v, theta):
        """Rotate 3-vector v about X by theta (right-hand rule)."""
        c = np.cos(theta); s = np.sin(theta)
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0,   c,  -s],
                      [0.0,   s,   c]])
        return R.dot(v)

    # ----------------------- planar 2-link IK -----------------------
    def planar_2link_ik(self, x, z, clamp_reach=True):
        """Solve planar 2-link IK in sagittal plane (x forward, z down).

        Returns (hip_pitch, knee_internal, reachable_flag).
        knee_internal is >0 when folded; 0 means straight leg.
        """
        l1 = self.l_thigh
        l2 = self.l_calf
        r = np.hypot(x, z)
        if clamp_reach:
            r_clamped = self.clamp(r, abs(l1 - l2) + 1e-8, l1 + l2 - 1e-8)
        else:
            r_clamped = r

        # law of cosines for knee internal angle
        cos_k = (l1 * l1 + l2 * l2 - r_clamped * r_clamped) / (2.0 * l1 * l2)
        cos_k = self.clamp(cos_k, -1.0, 1.0)
        knee_internal = np.pi - np.arccos(cos_k)

        # angle from hip to target (z downwards -> use -z)
        gamma = np.arctan2(-z, x)
        cos_a = (l1 * l1 + r_clamped * r_clamped - l2 * l2) / (2.0 * l1 * r_clamped)
        cos_a = self.clamp(cos_a, -1.0, 1.0)
        alpha = np.arccos(cos_a)
        hip_pitch = gamma - alpha

        reachable = abs(r - r_clamped) < 1e-6
        return hip_pitch, knee_internal, reachable

    # ----------------------- full 3-DOF IK -----------------------
    def full_leg_ik(self, x, y, z, clamp_reach=True):
        """Compute full 3-DOF IK for one leg.

        Parameters
        ----------
        x, y, z : float
            Target foot coordinates in hip frame (m). x forward, y left, z down.
        clamp_reach : bool
            If True, clamp unreachable targets to boundary.

        Returns
        -------
        IKResult: dataclass with fields (hip_abd, hip_pitch, knee, reachable)
        """
        # robust planar distance
        planar_dist = np.hypot(x, z)
        if planar_dist < self.EPS:
            planar_dist = self.EPS

        # hip abduction: limited atan2(y, planar_dist) to keep angle bounded
        hip_abd = self.hip_abd_sign * np.arctan2(y, planar_dist)

        # rotate foot into sagittal plane
        foot = np.array([x, y, z], dtype=float)
        foot_rot = self.rot_x_vec(foot, -hip_abd)
        x_plane, z_plane = float(foot_rot[0]), float(foot_rot[2])

        hip_pitch, knee_internal, reachable = self.planar_2link_ik(x_plane, z_plane, clamp_reach)

        # convert knee internal angle to motor calf joint (apply sign)
        calf_q = float(self.calf_sign * knee_internal)

        return IKResult(float(hip_abd), float(hip_pitch), calf_q, bool(reachable))

    # ----------------------- vertical-only IK -----------------------
    def vertical_only_ik(self, x0, y, z_desired, hip_pitch_nominal=None,
                         allow_small_pitch_adjust=False, clamp_reach=True, max_pitch_delta=0.15):
        """Constrained IK that preserves forward X coordinate x0 (prevents fore/aft motion).

        Strategy:
         - compute hip_abd to remove lateral y component
         - rotate into sagittal plane using hip_abd
         - solve planar IK for (x0, z_desired)
         - optionally lock hip_pitch to hip_pitch_nominal or allow a small adjustment
        """
        # ensure non-zero planar distance
        planar_dist = np.hypot(x0, z_desired)
        if planar_dist < self.EPS:
            planar_dist = self.EPS

        hip_abd = self.hip_abd_sign * np.arctan2(y, planar_dist)

        # rotate foot into sagittal plane
        foot = np.array([x0, y, z_desired], dtype=float)
        foot_rot = self.rot_x_vec(foot, -hip_abd)
        x_plane, z_plane = float(foot_rot[0]), float(foot_rot[2])

        hip_pitch_ik, knee_internal, reachable = self.planar_2link_ik(x_plane, z_plane, clamp_reach)
        calf_q = float(self.calf_sign * knee_internal)

        if hip_pitch_nominal is None:
            hip_pitch = float(hip_pitch_ik)
        else:
            if not allow_small_pitch_adjust:
                hip_pitch = float(hip_pitch_nominal)
            else:
                delta = hip_pitch_ik - float(hip_pitch_nominal)
                delta_clamped = self.clamp(delta, -abs(max_pitch_delta), abs(max_pitch_delta))
                hip_pitch = float(hip_pitch_nominal + delta_clamped)

        return IKResult(float(hip_abd), float(hip_pitch), calf_q, bool(reachable))






# ----------------------- example usage -----------------------
if __name__ == "__main__":
    ik = Go2LegIK(l_thigh=0.213, l_calf=0.213, hip_abd_sign=1.0, calf_sign=-1.0)

    # neutral foot for FR (example)
    neutral_fr = np.array([0.08, -0.09, -0.30])
    x0, y0, z0 = neutral_fr

    # move body up 0.02 m -> foot z becomes more negative
    d = 0.02
    z_new = z0 - d

    print("Full IK (may modify hip_pitch):")
    res_full = ik.full_leg_ik(x0, y0, z_new)
    print(res_full)

    print("Vertical-only IK (preserve x0, lock hip_pitch to stand):")
    stand_hip_pitch = 0.608813
    res_vert = ik.vertical_only_ik(x0, y0, z_new, hip_pitch_nominal=stand_hip_pitch,
                                   allow_small_pitch_adjust=False)
    print(res_vert)
