#!/usr/bin/env python3
"""
go2_walk_spline.py

Quadruped walking using cubic-spline foot trajectories + simple PID balance controller.
Requires: numpy, scipy, matplotlib (for spline plotting if used)
Uses your existing GO2_IK.Go2LegIK and unitree_sdk2py comms (as in original file).
"""

import time
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from GO2_IK import Go2LegIK
from spline_gen import SplineGenerator
from spline_gen import SplineGaitController

# ---------------------- constants / config ----------------------
DT          = 0.002
KP, KD      = 100.0, 2.0

NEUTRAL_FOOT = {
    'FR': np.array([ 0.08, -0.09, -0.35]),
    'FL': np.array([ 0.08,  0.09, -0.35]),
    'RR': np.array([-0.08, -0.09, -0.35]),
    'RL': np.array([-0.08,  0.09, -0.35]),
}

LEGS = ['FR','FL','RR','RL']


# ---------------------- helpers (unchanged) ----------------------
def get_current_motor_positions(state):
    if state is None:
        return np.zeros(12)
    positions = []
    motor_list = getattr(state, "motor_state", None) or getattr(state, "motorState", None) or []
    for i in range(min(12, len(motor_list))):
        m = motor_list[i]
        q = None
        for attr in ("q","q_target","q_actual","position"):
            if hasattr(m, attr):
                q = getattr(m, attr); break
        positions.append(float(q) if q is not None else 0.0)
    if len(positions) < 12:
        positions += [0.0] * (12 - len(positions))
    return np.array(positions)

def leg_indices(leg):
    idx = {'FR':0,'FL':1,'RR':2,'RL':3}[leg]
    return idx*3, idx*3+1, idx*3+2


# ---------------------- BalanceController (NEW) ----------------------
class BalanceController:
    """
    Simple PID-based balance controller that uses body roll & pitch to produce
    vertical foot offsets per leg to help stabilize the robot.
    - Reads roll/pitch and their rates from the LowState_ if available.
    - Returns per-leg z offsets (meters) that are added to foot targets.
    TUNING: start with conservative gains, increase slowly.
    """

    def __init__(self,
                 Kp_pitch=0.12, Ki_pitch=0.0, Kd_pitch=0.02,
                 Kp_roll=0.10, Ki_roll=0.0, Kd_roll=0.02,
                 max_offset=0.05):
        # pitch PID
        self.Kp_pitch = Kp_pitch
        self.Ki_pitch = Ki_pitch
        self.Kd_pitch = Kd_pitch
        self.pitch_int = 0.0

        # roll PID
        self.Kp_roll = Kp_roll
        self.Ki_roll = Ki_roll
        self.Kd_roll = Kd_roll
        self.roll_int = 0.0

        # state
        self.last_t = None
        self.last_pitch = 0.0
        self.last_roll = 0.0

        # clamp
        self.max_offset = float(max_offset)

        # mapping: legs to side/fore coefficients
        # front legs get fore=+1, rear legs fore=-1
        # left legs get side=+1, right legs side=-1
        self.leg_map = {
            'FR': {'fore':  1.0, 'side': -1.0},  # front-right: fore +, side -
            'FL': {'fore':  1.0, 'side':  1.0},  # front-left:  fore +, side +
            'RR': {'fore': -1.0, 'side': -1.0},
            'RL': {'fore': -1.0, 'side':  1.0},
        }

    def _extract_attitude(self, state):
        """
        Try multiple possible field names to obtain roll, pitch (radians) and
        their rates (rad/s). Returns (roll, pitch, roll_rate, pitch_rate).
        If nothing found, returns None.
        Common field shapes vary by SDK/version; check several names.
        """
        if state is None:
            return None

        # try common imu containers
        # 1) state.imu or state.imu[0] with .rpy or .rpy_rad or .angle / .gyro
        imu = getattr(state, 'imu', None)
        if imu:
            # imu could be a list
            if isinstance(imu, (list, tuple)) and len(imu) > 0:
                imu = imu[0]
            # try rpy
            rpy = getattr(imu, 'rpy', None) or getattr(imu, 'rpy_rad', None) or getattr(imu, 'angle', None)
            if rpy is not None:
                # rpy might be tuple/list
                try:
                    roll, pitch, yaw = list(rpy)[:3]
                except Exception:
                    roll = getattr(rpy, 'x', 0.0)
                    pitch = getattr(rpy, 'y', 0.0)
                # gyro rates
                gyro = getattr(imu, 'gyro', None) or getattr(imu, 'angular_velocity', None)
                if gyro is not None:
                    try:
                        gx, gy, gz = list(gyro)[:3]
                    except Exception:
                        gx = getattr(gyro, 'x', 0.0)
                        gy = getattr(gyro, 'y', 0.0)
                    # convention: assume gyro.x ~ roll rate, gyro.y ~ pitch rate
                    roll_rate = gx
                    pitch_rate = gy
                else:
                    roll_rate = 0.0
                    pitch_rate = 0.0
                return float(roll), float(pitch), float(roll_rate), float(pitch_rate)

        # 2) check for state.body_rpy, state.rpy, state.imu_rpy etc.
        for attr in ('body_rpy', 'rpy', 'imu_rpy', 'attitude', 'euler'):
            val = getattr(state, attr, None)
            if val is not None:
                try:
                    roll, pitch, yaw = list(val)[:3]
                    # try to find rate fields near attr (not guaranteed)
                    roll_rate = getattr(state, attr + '_rate', 0.0)
                    pitch_rate = getattr(state, attr + '_rate_y', 0.0) if hasattr(state, attr + '_rate_y') else 0.0
                    return float(roll), float(pitch), float(roll_rate), float(pitch_rate)
                except Exception:
                    continue

        # 3) nothing found
        return None

    def update(self, state, t):
        """
        Compute per-leg z offsets (meters) from current state and time t.
        Returns a dict: {leg: z_offset, ...}
        Positive z_offset -> less negative foot (foot raised), negative -> foot lowered.
        (Your neutral foot z values are negative; adding a negative offset lowers foot further)
        """
        att = self._extract_attitude(state)
        if att is None:
            # no attitude info; return zeros
            return {leg: 0.0 for leg in LEGS}

        roll, pitch, roll_rate, pitch_rate = att

        # dt
        if self.last_t is None:
            dt = DT
        else:
            dt = max(1e-6, t - self.last_t)

        # pitch PID
        pitch_err = pitch  # aim to drive pitch to 0
        self.pitch_int += pitch_err * dt
        # anti-windup clamp integral
        max_int = self.max_offset * 5.0
        self.pitch_int = np.clip(self.pitch_int, -max_int, max_int)
        pitch_d = (pitch - self.last_pitch) / dt if dt > 0 else 0.0
        pitch_out = (self.Kp_pitch * pitch_err +
                     self.Ki_pitch * self.pitch_int +
                     self.Kd_pitch * (-pitch_rate if pitch_rate is not None else -pitch_d))

        # roll PID
        roll_err = roll
        self.roll_int += roll_err * dt
        self.roll_int = np.clip(self.roll_int, -max_int, max_int)
        roll_d = (roll - self.last_roll) / dt if dt > 0 else 0.0
        roll_out = (self.Kp_roll * roll_err +
                    self.Ki_roll * self.roll_int +
                    self.Kd_roll * (-roll_rate if roll_rate is not None else -roll_d))

        # update history
        self.last_t = t
        self.last_pitch = pitch
        self.last_roll = roll

        # Convert to per-leg vertical offsets.
        # pitch_out: positive -> nose-up -> lower front legs (make z more negative) => we subtract pitch_out from front legs
        # roll_out: positive -> left-up -> lower left legs (make z more negative) => subtract roll_out from left legs
        offsets = {}
        for leg in LEGS:
            m = self.leg_map[leg]
            fore = m['fore']   # +1 front, -1 rear
            side = m['side']   # +1 left, -1 right

            # Contribution signs chosen so positive pitch_out lowers front legs
            z_from_pitch = -fore * pitch_out
            z_from_roll  = -side * roll_out

            z = z_from_pitch + z_from_roll

            # clamp
            z = float(np.clip(z, -self.max_offset, self.max_offset))
            offsets[leg] = z

        return offsets

# ---------------------- main loop ----------------------
def main():
    ChannelFactoryInitialize(1, 'lo')

    pub = ChannelPublisher('rt/lowcmd', LowCmd_)
    pub.Init()
    sub = ChannelSubscriber('rt/lowstate', LowState_)
    sub.Init()

    ik = Go2LegIK(l_thigh=0.213, l_calf=0.213, hip_abd_sign=1.0, calf_sign=-1.0)

    # gait and spline setup
    spline_gen = SplineGenerator()
    gait = SplineGaitController(NEUTRAL_FOOT, spline_gen,
                            step_length=0.18, step_height=0.1,
                            cycle_time=0.3, duty=0.6,
                            forward_dir=1.0)

    # balance controller (tune these gains carefully)
    bal = BalanceController(
        Kp_pitch=0.12, Ki_pitch=0.0, Kd_pitch=0.02,
        Kp_roll=0.10, Ki_roll=0.0, Kd_roll=0.02,
        max_offset=0.04
    )

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE; cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF; cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q    = 0.0
        cmd.motor_cmd[i].kp   = 0.0
        cmd.motor_cmd[i].dq   = 0.0
        cmd.motor_cmd[i].kd   = 0.0
        cmd.motor_cmd[i].tau  = 0.0

    crc = CRC()
    start_time = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            t = t0 - start_time

            # read state (best effort)
            try:
                state = sub.Read()
            except Exception:
                state = None
            curr_pos = get_current_motor_positions(state)

            # compute foot targets
            foot_targets = {}
            for leg in LEGS:
                foot_targets[leg] = gait.foot_target(leg, t)

            # balance: get per-leg z offsets and apply to foot targets BEFORE IK
            z_offsets = bal.update(state, t)
            for leg in LEGS:
                # add the offset to the foot z (neutral z values are negative)
                ft = foot_targets[leg].copy()
                ft[2] += z_offsets.get(leg, 0.0)
                foot_targets[leg] = ft

            # For each leg compute IK and fill cmd
            for idx, leg in enumerate(LEGS):
                hi, ti, ci = leg_indices(leg)
                ft = foot_targets[leg]  # (x,y,z) in hip frame
                res = ik.full_leg_ik(ft[0], ft[1], -ft[2])  # adjust sign as in your IK convention

                # The mapping below matches your original usage: hip_abd, hip_pitch, knee
                cmd.motor_cmd[hi].q = res.hip_abd
                cmd.motor_cmd[ti].q = -res.hip_pitch - np.pi/2
                cmd.motor_cmd[ci].q = res.knee

                cmd.motor_cmd[hi].kp = KP
                cmd.motor_cmd[hi].kd = KD
                cmd.motor_cmd[ti].kp = KP
                cmd.motor_cmd[ti].kd = KD
                cmd.motor_cmd[ci].kp = KP
                cmd.motor_cmd[ci].kd = KD

            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)

            # timing
            elapsed = time.perf_counter() - t0
            to_sleep = DT - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    except KeyboardInterrupt:
        print("Interrupted — stopping.")
    finally:
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        print("Done, exiting.")

if __name__ == '__main__':
    main()
