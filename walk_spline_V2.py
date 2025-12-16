import time
import sys
import numpy as np
from scipy.interpolate import CubicSpline

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


# --- Replace SplineTrajectoryGenerator and JointSpaceWalkController with this code ---

class SplineTrajectoryGenerator:
    """Generates smooth spline trajectories in joint space (returns spline objects)."""
    def __init__(self):
        # Tunable amplitudes (try small changes if robot still over/undershoots)
        self.lift_amplitude = 0.15
        self.forward_amplitude = 0.1

    def generate_swing_splines(self):
        """
        Create CubicSpline objects for lift and forward swing.
        Returns: (lift_spline, forward_spline)
        Each spline is defined on t in [0,1].
        """
        t = np.array([0.0, 0.3, 0.7, 1.0])

        # Lift: small at ends, larger in middle
        lift_points = np.array([0.0, 0.9, 0.9, 0.0])

        # Forward: start slightly behind, pass under body, end forward
        # Note: sign convention now chosen so positive forward_points means "forward swing".
        forward_points = np.array([-1.0, -0.3, 0.6, 1.0])

        lift_spline = CubicSpline(t, lift_points, bc_type='clamped')
        forward_spline = CubicSpline(t, forward_points, bc_type='clamped')

        return lift_spline, forward_spline


class JointSpaceWalkController:
    """
    Joint-space walking controller using hip for fore/aft motion.
    Adds small forward bias and stance downforce for traction.
    """
    def __init__(self):
        self.spline_gen = SplineTrajectoryGenerator()
        self.lift_spline, self.forward_spline = self.spline_gen.generate_swing_splines()

        # Gait timing
        self.gait_period = 1.0
        self.duty_factor = 0.60   # a bit longer stance for more traction

        # Standing angles [hip, thigh, calf]
        self.stand_angles = {
            'FR': np.array([0.00571868, 0.608813, -1.21763]),
            'FL': np.array([-0.00571868, 0.608813, -1.21763]),
            'RR': np.array([0.00571868, 0.608813, -1.21763]),
            'RL': np.array([-0.00571868, 0.608813, -1.21763])
        }

        # Diagonal trot
        self.phase_offsets = {
            'FR': 0.0,
            'RL': 0.0,
            'FL': 0.5,
            'RR': 0.5
        }

        # IMPORTANT: flip to match your hardware sign (this was the culprit).
        self.hip_direction = 1.0

        # Small biases to help overcome friction and stop sliding
        self.stride_bias = 0.035      # rad, added forward in swing, slight back in stance
        self.stance_downforce = 0.03  # rad added to thigh during stance (press foot)

    def get_joint_angles(self, leg_name, phase):
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        q = self.stand_angles[leg_name].copy()  # [hip, thigh, calf]

        if leg_phase < self.duty_factor:
            # ----- STANCE: foot on ground, push body forward -----
            s = leg_phase / self.duty_factor  # 0..1 across stance

            # Move foot backward relative to body over stance
            hip_offset = self.hip_direction * self.spline_gen.forward_amplitude * (0.5 - s)
            # small backward bias (negative of swing bias) to ensure net push
            hip_offset -= 0.5 * self.stride_bias

            q[0] += hip_offset

            # add a little downforce by increasing thigh (press into ground)
            # smooth bell curve over stance
            bell = 4.0 * s * (1.0 - s)  # 0 at ends, 1 at mid
            q[1] += self.stance_downforce * bell

        else:
            # ----- SWING: lift + move forward to place next step -----
            u = (leg_phase - self.duty_factor) / (1.0 - self.duty_factor)  # 0..1
            lift_val = float(self.lift_spline(u))
            fwd_val  = float(self.forward_spline(u))

            # Lift with thigh
            q[1] += self.spline_gen.lift_amplitude * lift_val

            # Swing hip forward (+ small bias to ensure foot lands ahead)
            hip_swing = self.hip_direction * self.spline_gen.forward_amplitude * fwd_val
            q[0] += hip_swing + self.stride_bias

        return q

    def compute_all_joints(self, phase):
        joint_positions = np.zeros(12)
        legs = ['FR', 'FL', 'RR', 'RL']
        for i, leg in enumerate(legs):
            joint_positions[i*3:(i+1)*3] = self.get_joint_angles(leg, phase)
        return joint_positions




class PoseTransition:
    """Smooth transition between poses"""
    
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
        
        # Smooth interpolation
        smooth_phase = 3*phase**2 - 2*phase**3  # Smoothstep
        
        MAX_DELTA_Q = 0.06  # rad per control step (tune down if still jerky)

        for i in range(12):
            desired_q = (smooth_phase * self.target_pos[i] +
                        (1 - smooth_phase) * self.beginning_pos[i])

            # clamp change relative to existing cmd to avoid jumps
            current_q = getattr(cmd.motor_cmd[i], "q", 0.0)
            delta = desired_q - current_q
            if delta > MAX_DELTA_Q: 
                desired_q = current_q + MAX_DELTA_Q
            elif delta < -MAX_DELTA_Q:
                desired_q = current_q - MAX_DELTA_Q

            cmd.motor_cmd[i].q = desired_q
            # much lower gains during big pose moves to avoid violent torques
            cmd.motor_cmd[i].kp = 80.0
            cmd.motor_cmd[i].kd = 6.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0

        if phase >= 1.0:
            self.active = False
            return True
        return False


def get_current_motor_positions(state):
    """Extract current motor positions from state message"""
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
    
    return np.array(positions)



def quaternion_to_euler(qw, qx, qy, qz):
    """
    Convert quaternion (w,x,y,z) to roll, pitch, yaw (radians).
    """
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_body_orientation(state):
    """
    Read IMU orientation from LowState_ message.
    Returns (roll, pitch, yaw) in radians.
    Tries several common field names; returns (0,0,0) if not available.
    """
    if state is None:
        return 0.0, 0.0, 0.0

    imu = getattr(state, "imu", None) or getattr(state, "IMU", None) or None
    if imu is None:
        # try nested or alternate names
        for attrname in dir(state):
            if "imu" in attrname.lower():
                imu = getattr(state, attrname, None)
                break

    if imu is None:
        return 0.0, 0.0, 0.0

    # Try quaternion first: (w,x,y,z) common
    for qname in ("quaternion", "q", "quat", "orientation"):
        q = getattr(imu, qname, None)
        if q is not None:
            # detect shape: could be list/tuple/array or object with fields
            try:
                if len(q) >= 4:
                    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                    return quaternion_to_euler(qw, qx, qy, qz)
            except Exception:
                # object with fields x,y,z,w
                try:
                    qw = float(getattr(q, "w", q[0]))
                    qx = float(getattr(q, "x", q[1]))
                    qy = float(getattr(q, "y", q[2]))
                    qz = float(getattr(q, "z", q[3]))
                    return quaternion_to_euler(qw, qx, qy, qz)
                except Exception:
                    pass

    # Try rpy / euler fields
    for name in ("rpy", "euler", "eulerAngle", "rpy_rad"):
        val = getattr(imu, name, None)
        if val is not None:
            try:
                # list-like or object
                if len(val) >= 3:
                    return float(val[0]), float(val[1]), float(val[2])
            except Exception:
                # object with roll/pitch/yaw fields
                r = getattr(val, "roll", None) or getattr(val, "x", None)
                p = getattr(val, "pitch", None) or getattr(val, "y", None)
                y = getattr(val, "yaw", None) or getattr(val, "z", None)
                if r is not None and p is not None and y is not None:
                    return float(r), float(p), float(y)

    # nothing found
    return 0.0, 0.0, 0.0


class PIDController:
    """Simple PID with integrator clamp and derivative on measurement"""
    def __init__(self, kp=0.8, ki=0.0, kd=0.02, integrator_limit=0.5, output_limit=0.4):
        self.kp = kp
        self.ki = ki
        self.kd = kd
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
        # integral with simple anti-windup clamp
        self.integral += error * dt
        if self.integral > self.integrator_limit:
            self.integral = self.integrator_limit
        elif self.integral < -self.integrator_limit:
            self.integral = -self.integrator_limit

        # derivative
        derivative = 0.0
        if self.prev_error is not None:
            derivative = (error - self.prev_error) / dt

        self.prev_error = error

        out = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        # clamp output
        if out > self.output_limit:
            out = self.output_limit
        elif out < -self.output_limit:
            out = -self.output_limit
        return out


def main():
    # Initial poses
    stand_up_pos = np.array([
        0.00571868, 0.608813, -1.21763,  # FR
        -0.00571868, 0.608813, -1.21763,  # FL
        0.00571868, 0.608813, -1.21763,   # RR
        -0.00571868, 0.608813, -1.21763   # RL
    ])
    
    stand_down_pos = np.array([
        0.0473455, 1.22187, -2.44375,
        -0.0473455, 1.22187, -2.44375,
        0.0473455, 1.22187, -2.44375,
        -0.0473455, 1.22187, -2.44375
    ])
    
    # Initialize SDK
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])
    
    # Create publisher and subscriber
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    
    # Initialize command
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
    
    # Controllers
    pose_transition = PoseTransition()
    walk_controller = JointSpaceWalkController()
    crc = CRC()

    # Balance PID: tune these gains for your robot
    # Start conservative: low kp, small output_limit (radians)
    balance_pid = PIDController(kp=0.9, ki=0.05, kd=0.03,
                                integrator_limit=0.2, output_limit=0.12)
    desired_pitch = 0.0   # radians: target body pitch (0 = level)
    
    # Timing
    dt = 0.002
    running_time = 0.0
    
    # State machine
    state_machine = "IDLE"
    walk_start_time = 0.0
    walk_duration = 6.0  # Walk for 6 seconds
    
    input("Press Enter to start standing up, then walking...")
    
    while True:
        step_start = time.perf_counter()
        running_time += dt
        
        # Read current state
        try:
            state = sub.Read()
        except Exception as e:
            state = None
        
        current_pos = get_current_motor_positions(state)
        
        # State machine
        if state_machine == "IDLE":
            if running_time > 0.5:
                print("Standing up...")
                pose_transition.start(current_pos, stand_up_pos, running_time, 1.5)
                state_machine = "STANDING_UP"
        
        elif state_machine == "STANDING_UP":
            finished = pose_transition.update(cmd, running_time)
            if finished:
                print("Standing complete. Starting walk in 1 second...")
                state_machine = "PRE_WALK"
                walk_start_time = running_time + 1.0
        
        elif state_machine == "PRE_WALK":
            # Hold standing position
            for i in range(12):
                cmd.motor_cmd[i].q = stand_up_pos[i]
                cmd.motor_cmd[i].kp = 50.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 3.5
                cmd.motor_cmd[i].tau = 0.0
            
            if running_time >= walk_start_time:
                print("Starting walk with spline trajectories...")
                state_machine = "WALKING"
        
        elif state_machine == "WALKING":
            walk_time = running_time - walk_start_time

            if walk_time < walk_duration:
                # Calculate gait phase
                phase = (walk_time / walk_controller.gait_period) % 1.0

                # Compute joint angles using spline trajectories
                target_joints = walk_controller.compute_all_joints(phase)

                # --- Balance PID setup (ensure you created balance_pid earlier in main)
                # If you didn't, create one above main loop:
                # balance_pid = PIDController(kp=0.9, ki=0.05, kd=0.03, integrator_limit=0.2, output_limit=0.12)
                # desired_pitch = 0.0
                try:
                    roll, pitch, yaw = get_body_orientation(state)
                except Exception:
                    roll, pitch, yaw = 0.0, 0.0, 0.0

                # Pitch error: positive means nose-up. We want to hold desired_pitch (0 = level)
                pitch_error = (desired_pitch if 'desired_pitch' in globals() else 0.0) - pitch

                # update PID (use dt from your timing)
                pid_dt = dt if 'dt' in globals() else 0.002
                pid_output = balance_pid.update(pitch_error, pid_dt)

                # Scale PID and distribute across hips
                pid_scale = 0.7  # tune this between 0.3..1.0
                corrected_targets = target_joints.copy()
                hip_indices = [0, 3, 6, 9]
                for hi in hip_indices:
                    corrected_targets[hi] += pid_scale * pid_output

                # Diagnostics: show intended hip offsets (indices 0,3,6,9) and PID info
                hips = [corrected_targets[i] for i in hip_indices]
                mean_hip = sum(hips) / 4.0
                if int(walk_time * 10) % 10 == 0:
                    print(f"DBG phase {phase:.2f} hips(FR,FL,RR,RL): " +
                          ", ".join(f"{h:.3f}" for h in hips) +
                          f"  mean_hip={mean_hip:.3f}  pitch={pitch:.3f} pid_out={pid_output:.4f}")

                # Apply with rate limiting to prevent jumps (same limiter you used)
                MAX_DELTA_Q_WALK = 0.04
                for i in range(12):
                    desired_q = corrected_targets[i]
                    current_q = getattr(cmd.motor_cmd[i], "q", 0.0)
                    delta = desired_q - current_q
                    if delta > MAX_DELTA_Q_WALK:
                        desired_q = current_q + MAX_DELTA_Q_WALK
                    elif delta < -MAX_DELTA_Q_WALK:
                        desired_q = current_q - MAX_DELTA_Q_WALK

                    cmd.motor_cmd[i].q = desired_q
                    # Stiffer but damped for walking
                    cmd.motor_cmd[i].kp = 280.0
                    cmd.motor_cmd[i].kd = 10.0
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
            pass
        
        # Send command
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        
        # Maintain timing
        time_until_next = dt - (time.perf_counter() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)


if __name__ == '__main__':
    main()