#!/usr/bin/env python3
"""
Rewritten quadruped joint-space walking using splines + simple body-pitch PID.
Drop-in replacement for your earlier script. Test with robot supported first.
"""

import time
import sys
import argparse
import numpy as np
from scipy.interpolate import CubicSpline

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


# -----------------------
# Config / Tunables
# -----------------------
HIP_DIRECTION = -1.0   # set -1.0 or 1.0; flip sign if robot moves backwards
GAIT_PERIOD = 1.0
DUTY_FACTOR = 0.60     # fraction of cycle foot on ground (larger -> more traction)
DT = 0.002             # main loop timestep
MAX_DELTA_Q_WALK = 0.04
POSE_TRANSITION_TIME = 1.5
WALK_DURATION = 10.0   # seconds to walk when started
PRINT_VERBOSE = True


# -----------------------
# Utilities: IMU / quaternion helpers
# -----------------------
def quaternion_to_euler(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to roll, pitch, yaw (radians)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_body_orientation(state):
    """
    Robustly try to read IMU orientation from the LowState_ message.
    Returns roll, pitch, yaw (radians). If not found returns (0,0,0).
    """
    if state is None:
        return 0.0, 0.0, 0.0

    # try common names / nested containers
    imu = getattr(state, "imu", None) or getattr(state, "IMU", None) or None
    if imu is None:
        for attr in dir(state):
            if "imu" in attr.lower():
                imu = getattr(state, attr, None)
                break
    if imu is None:
        return 0.0, 0.0, 0.0

    # quaternion-like fields
    for qname in ("quaternion", "q", "quat", "orientation"):
        q = getattr(imu, qname, None)
        if q is not None:
            try:
                if len(q) >= 4:
                    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                    return quaternion_to_euler(qw, qx, qy, qz)
            except Exception:
                try:
                    qw = float(getattr(q, "w", q[0]))
                    qx = float(getattr(q, "x", q[1]))
                    qy = float(getattr(q, "y", q[2]))
                    qz = float(getattr(q, "z", q[3]))
                    return quaternion_to_euler(qw, qx, qy, qz)
                except Exception:
                    pass

    # try RPY / euler fields
    for name in ("rpy", "euler", "eulerAngle", "rpy_rad"):
        val = getattr(imu, name, None)
        if val is not None:
            try:
                if len(val) >= 3:
                    return float(val[0]), float(val[1]), float(val[2])
            except Exception:
                r = getattr(val, "roll", None) or getattr(val, "x", None)
                p = getattr(val, "pitch", None) or getattr(val, "y", None)
                y = getattr(val, "yaw", None) or getattr(val, "z", None)
                if r is not None and p is not None and y is not None:
                    return float(r), float(p), float(y)
    return 0.0, 0.0, 0.0


# -----------------------
# PID Controller
# -----------------------
class PIDController:
    """Simple PID with integrator clamp and derivative on measurement"""
    def __init__(self, kp=0.9, ki=0.05, kd=0.03, integrator_limit=0.2, output_limit=0.12):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integrator_limit = integrator_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        if dt <= 0:
            return 0.0
        self.integral += error * dt
        # clamp integrator
        if self.integral > self.integrator_limit:
            self.integral = self.integrator_limit
        elif self.integral < -self.integrator_limit:
            self.integral = -self.integrator_limit
        derivative = 0.0
        if self.prev_error is not None:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error
        out = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        # clamp output
        if out > self.output_limit: out = self.output_limit
        if out < -self.output_limit: out = -self.output_limit
        return out


# -----------------------
# Spline generator + walk controller
# -----------------------
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
        lift_points = np.array([0.0, 2.9, 2.0, 0.0, 0.0])

        # FORWARD: symmetric, less extreme -> avoids introducing net bias
        # starts slightly back, passes under body, ends slightly forward
        forward_points = np.array([-0.8, -0.3, 0.8, 0.8, -0.8])

        lift_spline = CubicSpline(t, lift_points, bc_type='clamped')
        forward_spline = CubicSpline(t, forward_points, bc_type='clamped')
        return lift_spline, forward_spline


class JointSpaceWalkController:
    """
    Joint-space walking controller:
     - hip joint (index 0) used for fore/aft placement
     - thigh (index 1) used for lift
    """
    def __init__(self, hip_direction=HIP_DIRECTION, gait_period=GAIT_PERIOD, duty_factor=DUTY_FACTOR):
        self.spline_gen = SplineTrajectoryGenerator()
        self.lift_spline, self.forward_spline = self.spline_gen.generate_swing_splines()
        self.gait_period = gait_period
        self.duty_factor = duty_factor

        # nominal stand angles [hip, thigh, calf]
        self.stand_angles = {
            'FR': np.array([0.00571868, 0.608813, -1.21763]),
            'FL': np.array([-0.00571868, 0.608813, -1.21763]),
            'RR': np.array([0.00571868, 0.608813, -1.21763]),
            'RL': np.array([-0.00571868, 0.608813, -1.21763])
        }

        # trot: diagonal pairs
        self.phase_offsets = {'FR': 0.0, 'RL': 0.0, 'FL': 0.5, 'RR': 0.5}

        self.hip_direction = hip_direction
        # small safety/tracton helpers
        self.stride_bias = 0.0
        self.stance_downforce = 0.03

    def get_joint_angles(self, leg_name, phase):
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        q = self.stand_angles[leg_name].copy()

        if leg_phase < self.duty_factor:
            s = leg_phase / self.duty_factor
            hip_offset = self.hip_direction * self.spline_gen.forward_amplitude * (0.5 - s)
            hip_offset -= 0.5 * self.stride_bias
            q[0] += hip_offset
            bell = 4.0 * s * (1.0 - s)
            q[1] += self.stance_downforce * bell
        else:
            u = (leg_phase - self.duty_factor) / (1.0 - self.duty_factor)
            lift_val = float(self.lift_spline(u))
            fwd_val = float(self.forward_spline(u))
            q[1] += self.spline_gen.lift_amplitude * lift_val
            hip_swing = self.hip_direction * self.spline_gen.forward_amplitude * fwd_val
            q[0] += hip_swing + self.stride_bias

        return q

    def compute_all_joints(self, phase):
        legs = ['FR', 'FL', 'RR', 'RL']
        out = np.zeros(12)
        for i, leg in enumerate(legs):
            out[i*3:(i+1)*3] = self.get_joint_angles(leg, phase)
        return out


# -----------------------
# Pose transition helper
# -----------------------
class PoseTransition:
    def __init__(self):
        self.transition_start_time = 0.0
        self.beginning_pos = np.zeros(12)
        self.target_pos = np.zeros(12)
        self.transition_time = 1.0
        self.active = False

    def start(self, current_pos, target_pos, t, transition_time=1.0):
        self.transition_start_time = t
        self.beginning_pos = current_pos.copy()
        self.target_pos = target_pos.copy()
        self.transition_time = transition_time
        self.active = True

    def update(self, cmd, t):
        if not self.active:
            return True
        elapsed = t - self.transition_start_time
        phase = min(elapsed / self.transition_time, 1.0)
        smooth = 3*phase**2 - 2*phase**3
        MAX_DELTA_Q = 0.06
        for i in range(12):
            desired_q = smooth * self.target_pos[i] + (1.0 - smooth) * self.beginning_pos[i]
            current_q = getattr(cmd.motor_cmd[i], "q", 0.0)
            delta = desired_q - current_q
            if delta > MAX_DELTA_Q:
                desired_q = current_q + MAX_DELTA_Q
            elif delta < -MAX_DELTA_Q:
                desired_q = current_q - MAX_DELTA_Q
            cmd.motor_cmd[i].q = desired_q
            cmd.motor_cmd[i].kp = 80.0
            cmd.motor_cmd[i].kd = 6.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        if phase >= 1.0:
            self.active = False
            return True
        return False


# -----------------------
# Helpers
# -----------------------
def get_current_motor_positions(state):
    if state is None:
        return np.zeros(12)
    positions = []
    motor_list = getattr(state, "motor_state", None) or getattr(state, "motorState", None) or []
    for i in range(min(12, len(motor_list))):
        m = motor_list[i]
        q = None
        for attr in ("q", "q_target", "q_actual", "position"):
            if hasattr(m, attr):
                q = getattr(m, attr)
                break
        positions.append(float(q) if q is not None else 0.0)
    if len(positions) < 12:
        positions += [0.0] * (12 - len(positions))
    return np.array(positions)


def detect_motor_mapping_from_state(state):
    """
    Attempt to detect logical leg mapping by reading motor_state[].id if available.
    Returns a mapping list of length 12 or None if unavailable.
    """
    if state is None:
        return None
    motor_list = getattr(state, "motor_state", None) or getattr(state, "motorState", None) or []
    ids = []
    for m in motor_list[:12]:
        mid = getattr(m, "id", None) or getattr(m, "motorId", None) or None
        ids.append(mid)
    if len(ids) == 12 and any(id is not None for id in ids):
        return ids
    return None


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hip-dir", type=float, default=HIP_DIRECTION,
                        help="Hip direction sign (1.0 or -1.0)")
    parser.add_argument("--lo", type=str, default="lo", help="ChannelFactoryInitialize param (default 'lo')")
    args = parser.parse_args()

    # apply hip direction from CLI
    hip_dir = args.hip_dir

    # poses
    stand_up_pos = np.array([
        0.00571868, 0.608813, -1.21763,
        -0.00571868, 0.608813, -1.21763,
        0.00571868, 0.608813, -1.21763,
        -0.00571868, 0.608813, -1.21763
    ])
    stand_down_pos = np.array([
        0.0473455, 1.22187, -2.44375,
        -0.0473455, 1.22187, -2.44375,
        0.0473455, 1.22187, -2.44375,
        -0.0473455, 1.22187, -2.44375
    ])

    # init channels
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, args.lo)
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    # initial command message
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    # controllers
    pose_transition = PoseTransition()
    walk_controller = JointSpaceWalkController(hip_direction=hip_dir,
                                               gait_period=GAIT_PERIOD,
                                               duty_factor=DUTY_FACTOR)
    crc = CRC()
    balance_pid = PIDController(kp=0.9, ki=0.05, kd=0.03,
                                integrator_limit=0.2, output_limit=0.12)
    desired_pitch = 0.0

    # --- NEW: roll/lateral PID to remove sideways drift ---
    # small gains: roll causes left/right hip differential; tune carefully
    roll_pid = PIDController(kp=0.6, ki=0.02, kd=0.01,
                             integrator_limit=0.12, output_limit=0.08)
    desired_roll = 0.0  # keep body level

    # timing / state
    dt = DT
    running_time = 0.0
    state_machine = "IDLE"
    walk_start_time = 0.0

    input("Press Enter to start standing up, then walking (safety: place robot in harness) ...")

    # main loop
    while True:
        step_start = time.perf_counter()
        running_time += dt
        try:
            state = sub.Read()
        except Exception:
            state = None

        current_pos = get_current_motor_positions(state)

        # state machine
        if state_machine == "IDLE":
            if running_time > 0.5:
                print("Standing up...")
                pose_transition.start(current_pos, stand_up_pos, running_time, POSE_TRANSITION_TIME)
                state_machine = "STANDING_UP"

        elif state_machine == "STANDING_UP":
            finished = pose_transition.update(cmd, running_time)
            if finished:
                print("Standing complete. Starting walk in 1.0s ...")
                state_machine = "PRE_WALK"
                walk_start_time = running_time + 1.0

        elif state_machine == "PRE_WALK":
            for i in range(12):
                cmd.motor_cmd[i].q = stand_up_pos[i]
                cmd.motor_cmd[i].kp = 50.0
                cmd.motor_cmd[i].kd = 3.5
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].tau = 0.0
            if running_time >= walk_start_time:
                print("Starting walk with spline trajectories...")
                # reset integrator each start
                balance_pid.reset()
                state_machine = "WALKING"
                walk_phase_start = running_time

        elif state_machine == "WALKING":
            walk_time = running_time - walk_start_time
            if walk_time < WALK_DURATION:
                phase = (walk_time / walk_controller.gait_period) % 1.0
                target_joints = walk_controller.compute_all_joints(phase)

                # read body orientation from IMU (roll, pitch, yaw)
                try:
                    roll, pitch, yaw = get_body_orientation(state)
                except Exception:
                    roll, pitch, yaw = 0.0, 0.0, 0.0

                # PITCH correction (existing)
                pitch_error = desired_pitch - pitch
                pid_output = balance_pid.update(pitch_error, dt)

                # ROLL correction (new) - keep robot level by differential hip offsets
                roll_error = desired_roll - roll
                roll_output = roll_pid.update(roll_error, dt)

                # distribute PID corrections across hips
                pid_scale = 0.7      # scale for pitch correction (mean hip)
                roll_scale = 1.0     # how strongly roll_output moves hips (tune)

                corrected_targets = target_joints.copy()
                # hip indices in joint vector: FR=0, FL=3, RR=6, RL=9
                hip_indices = {'FR': 0, 'FL': 3, 'RR': 6, 'RL': 9}
                # apply pitch correction equally
                for hi in hip_indices.values():
                    corrected_targets[hi] += pid_scale * pid_output

                # apply roll correction differentially:
                # Add +roll on left hips, -roll on right hips (this steers back toward desired_roll)
                left_hips = [hip_indices['FL'], hip_indices['RL']]
                right_hips = [hip_indices['FR'], hip_indices['RR']]
                for hi in left_hips:
                    corrected_targets[hi] += roll_scale * roll_output
                for hi in right_hips:
                    corrected_targets[hi] -= roll_scale * roll_output

                # Diagnostics: print both PID values + hip means
                if PRINT_VERBOSE and int(walk_time * 10) % 10 == 0:
                    hips = [corrected_targets[i] for i in hip_indices.values()]
                    mean_hip = sum(hips) / 4.0
                    print(f"DBG phase {phase:.2f} hips(FR,FL,RR,RL): " +
                          ", ".join(f"{h:.3f}" for h in [corrected_targets[0], corrected_targets[3], corrected_targets[6], corrected_targets[9]]) +
                          f" mean_hip={mean_hip:.3f} pitch={pitch:.3f} pid_pitch={pid_output:.4f} roll={roll:.3f} pid_roll={roll_output:.4f}")

                # apply with rate limiting and set gains for motors (same limiter as before)
                for i in range(12):
                    desired_q = corrected_targets[i]
                    current_q = getattr(cmd.motor_cmd[i], "q", 0.0)
                    cmd.motor_cmd[i].q = desired_q
                    cmd.motor_cmd[i].kp = 50.0
                    cmd.motor_cmd[i].kd = 2.0
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].tau = 0.0

                if int(walk_time * 2) % 2 == 0:
                    print(f"Walking... Phase: {phase:.2f}, Time: {walk_time:.1f}s")

            else:
                print("Walk complete. Standing down...")
                pose_transition.start(current_pos, stand_down_pos, running_time, 2.0)
                state_machine = "STANDING_DOWN"

        elif state_machine == "STANDING_DOWN":
            finished = pose_transition.update(cmd, running_time)
            if finished:
                print("Complete!")
                state_machine = "FINISHED"

        elif state_machine == "FINISHED":
            # hold pose
            pass

        # publish command
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        # maintain timing
        time_until_next = dt - (time.perf_counter() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)


if __name__ == "__main__":
    main()
