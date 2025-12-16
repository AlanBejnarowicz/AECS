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


class SplineTrajectoryGenerator:
    """Generates smooth spline trajectories in joint space"""
    
    def __init__(self):
        self.lift_amplitude = 0.15  # How much to change thigh angle for lift
        self.forward_amplitude = 0.12  # How much to change calf angle for forward/back
        
    def generate_swing_spline(self, num_points=20):
        """
        Generate a smooth spline for the swing phase
        Returns: (lift_trajectory, forward_trajectory)
        """
        # Time points for spline
        t = np.array([0.0, 0.3, 0.7, 1.0])
        
        # Lift trajectory (for thigh joint)
        # Start low, lift up in middle, end low
        lift_points = np.array([0.0, 0.8, 0.8, 0.0])
        
        # Forward trajectory (for moving leg forward)
        # Start back, move forward
        forward_points = np.array([-1.0, -0.5, 0.5, 1.0])
        
        # Create cubic splines
        lift_spline = CubicSpline(t, lift_points, bc_type='clamped')
        forward_spline = CubicSpline(t, forward_points, bc_type='clamped')
        
        # Generate smooth trajectory
        t_interp = np.linspace(0, 1, num_points)
        
        return lift_spline(t_interp), forward_spline(t_interp)


class JointSpaceWalkController:
    """
    Walking controller that works directly in joint space
    Modulates joint angles around the known working stand position
    """
    
    def __init__(self):
        self.spline_gen = SplineTrajectoryGenerator()
        
        # Pre-generate spline trajectories
        self.lift_traj, self.forward_traj = self.spline_gen.generate_swing_spline()
        
        # Walking parameters
        self.gait_period = 1.0  # Full gait cycle time (seconds)
        self.duty_factor = 0.6  # Fraction of time foot is on ground
        
        # Base standing joint angles (from your working stand_up position)
        # [hip, thigh, calf] for each leg
        self.stand_angles = {
            'FR': np.array([0.00571868, 0.608813, -1.21763]),
            'FL': np.array([-0.00571868, 0.608813, -1.21763]),
            'RR': np.array([0.00571868, 0.608813, -1.21763]),
            'RL': np.array([-0.00571868, 0.608813, -1.21763])
        }
        
        # Phase offsets for trot gait (diagonal legs move together)
        self.phase_offsets = {
            'FR': 0.0,
            'RL': 0.0,
            'FL': 0.5,
            'RR': 0.5
        }
        
    def get_joint_angles(self, leg_name, phase):
        """
        Get joint angles for a leg at given gait phase
        Returns: [hip, thigh, calf] angles
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        base_angles = self.stand_angles[leg_name].copy()
        
        if leg_phase < self.duty_factor:
            # Stance phase - leg pushes backward
            stance_progress = leg_phase / self.duty_factor
            
            # Smoothly move leg from forward to backward
            forward_offset = self.spline_gen.forward_amplitude * (0.5 - stance_progress)
            
            # Modify calf joint for forward/backward (this moves foot fwd/back)
            base_angles[2] += forward_offset
            
        else:
            # Swing phase - lift leg and swing forward using spline
            swing_progress = (leg_phase - self.duty_factor) / (1.0 - self.duty_factor)
            
            # Get trajectory values from spline
            traj_idx = int(swing_progress * (len(self.lift_traj) - 1))
            lift_val = self.lift_traj[traj_idx]
            forward_val = self.forward_traj[traj_idx]
            
            # Apply to joints
            # Thigh lifts the leg up
            base_angles[1] += self.spline_gen.lift_amplitude * lift_val
            
            # Calf moves leg forward/backward
            base_angles[2] += self.spline_gen.forward_amplitude * forward_val * 0.5
            
        return base_angles
    
    def compute_all_joints(self, phase):
        """Compute all 12 joint positions for current gait phase"""
        joint_positions = np.zeros(12)
        
        legs = ['FR', 'FL', 'RR', 'RL']
        for i, leg in enumerate(legs):
            angles = self.get_joint_angles(leg, phase)
            joint_positions[i*3:(i+1)*3] = angles
            
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
        
        for i in range(12):
            cmd.motor_cmd[i].q = (smooth_phase * self.target_pos[i] + 
                                  (1 - smooth_phase) * self.beginning_pos[i])
            cmd.motor_cmd[i].kp = 50.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 3.5
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
                
                # Apply to motors
                for i in range(12):
                    cmd.motor_cmd[i].q = target_joints[i]
                    cmd.motor_cmd[i].kp = 35.0  # Moderate stiffness
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = 3.5
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