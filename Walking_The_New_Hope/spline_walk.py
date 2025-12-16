#!/usr/bin/env python3
"""
go2_walk_spline.py

Quadruped walking using cubic-spline foot trajectories.
Requires: numpy, scipy, matplotlib (for plotting)
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
    'FR': np.array([ 0.08, -0.09, -0.40]),
    'FL': np.array([ 0.08,  0.09, -0.40]),
    'RR': np.array([-0.08, -0.09, -0.40]),
    'RL': np.array([-0.08,  0.09, -0.40]),
}

LEGS = ['FR','FL','RR','RL']

# phase offsets for trot: FR & RL in-phase, FL & RR out-of-phase
PHASE_OFF = {'FR':0.0, 'RL':0.0, 'FL':0.5, 'RR':0.5}

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



# ---------------------- walking controller ----------------------
class SplineGaitController:
    """
    High-level gait controller producing foot targets for each leg over time.
    Added `forward_dir` to control walking direction: +1 = as before, -1 = reversed.
    """
    def __init__(self, neutral_foot_map, spline_gen: SplineGenerator,
                 step_length=0.22, step_height=0.12, cycle_time=0.6, duty=0.6,
                 forward_dir=1.0):
        self.neutral = neutral_foot_map
        self.sg = spline_gen
        self.step_length = float(step_length)
        self.step_height = float(step_height)
        self.cycle_time = float(cycle_time)
        self.duty = float(duty)  # fraction of cycle leg is on ground (stance)
        self.swing_time = self.cycle_time * (1.0 - self.duty)
        self.stance_time = self.cycle_time * self.duty

        # direction multiplier: set to -1 to walk the other way
        self.forward_dir = float(np.sign(forward_dir) if forward_dir != 0 else 1.0)

        # Precompute half-step offsets (foot positions relative to neutral).
        # Stance moves foot backward relative to body by default; multiply by forward_dir to flip.
        half = self.step_length / 2.0
        self.stance_start_offset = np.array([ half * self.forward_dir, 0.0, 0.0])
        self.stance_end_offset   = np.array([-half * self.forward_dir, 0.0, 0.0])

    def phase_of_time(self, t, offset):
        cycle_pos = ((t / self.cycle_time) + offset) % 1.0
        return cycle_pos

    def foot_target(self, leg, t):
        offset = PHASE_OFF[leg]
        phase = self.phase_of_time(t, offset)

        if phase < self.duty:
            s = phase / self.duty
            delta = (1.0 - s) * self.stance_start_offset + s * self.stance_end_offset
            target = self.neutral[leg] + delta
            return target
        else:
            s = (phase - self.duty) / (1.0 - self.duty)
            p_start = self.neutral[leg] + self.stance_end_offset
            p_end   = self.neutral[leg] + self.stance_start_offset
            spline = self.sg.plan_swing(p_start, p_end, lift=self.step_height)
            return spline(s)











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
                            cycle_time=0.6, duty=0.6,
                            forward_dir=1.0)  # flip walking direction

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

            # For each leg compute IK and fill cmd
            for idx, leg in enumerate(LEGS):
                hi, ti, ci = leg_indices(leg)
                ft = foot_targets[leg]  # (x,y,z) in hip frame (we assume neutral is hip-frame)
                res = ik.full_leg_ik(ft[0], ft[1], -ft[2])  # note your IK might expect z positive down/up -> adjust

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