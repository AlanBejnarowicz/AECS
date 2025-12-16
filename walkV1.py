import time
import sys
import numpy as np



from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_



class PoseTransition:
    def __init__(self):
        self.transition_start_time = 0.0
        self.begining_pos = np.zeros(12)
        self.last_transition_finished = True
        self.target_pos = np.zeros(12)
        self.trasition_time = 1.0

    def prepare_for_new_transition(self, current_pos, target_pos, t, transition_time): 
        self.transition_start_time = t
        self.begining_pos = current_pos.copy()
        self.last_transition_finished = False
        self.target_pos = target_pos.copy()
        self.transition_time = transition_time
    



    def transition_to_pose(self, cmd, t):

        print(f"DEBUG   ---     Time: {t:.2f}s")
        print(f"DEBUG   ---     Transition start time: {self.transition_start_time:.2f}s")
        
        trans_t = t - self.transition_start_time
        print(f"DEBUG   ---     Transition time elapsed: {trans_t:.2f}s / {self.transition_time:.2f}s")

        phase = min((t - self.transition_start_time) / self.transition_time, 1.0)
        #phase = np.tanh( (trans_t / self.transition_time) * np.pi )

        print(f"DEBUG   ---     Transition phase: {phase:.2f}")

        for i in range(12):
            cmd.motor_cmd[i].q = phase * self.target_pos[i] + (
                1 - phase) * self.begining_pos[i]
            cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 3.5
            cmd.motor_cmd[i].tau = 0.0

        if phase >= 1.0:
            print("DEBUG   ---     Pose transition complete.")
            return True
        
        return False
            

PS = PoseTransition()

stand_up_joint_pos = np.array([
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
],
                              dtype=float)

stand_down_joint_pos = np.array([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
],
                                dtype=float)






def get_current_motor_positions(state):
    if state is None:
        print("ERROR    ---     State is None, cannot get motor positions.")
        return None
    
    positions = []
    # try a few common field name variants to be robust
    motor_list = getattr(state, "motor_state", None) or getattr(state, "motorState", None) or []
    for i in range(min(12, len(motor_list))):
        m = motor_list[i]
        q = None
        for attr in ("q", "q_target", "q_actual", "position"):
            if hasattr(m, attr):
                q = getattr(m, attr)
                break
        positions.append(float(q) if q is not None else None)
    return positions



dt = 0.002
runing_time = 0.0
crc = CRC()

input("Press enter to start")

if __name__ == '__main__':

    if len(sys.argv) <2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    
    operation_finished = True
    while True:
        step_start = time.perf_counter()
        runing_time += dt


            # read motor positions

        try:
            state = sub.Read()  # may return None if no message yet
        except Exception as e:
            print("Failed to read motor positions:", e)
            state = None

        motors_readout = get_current_motor_positions(state)




        #stand up
        if (runing_time > 0.5 and runing_time <= 0.5 + 1.2):
            if(operation_finished == True):
                PS.prepare_for_new_transition(motors_readout, stand_up_joint_pos,
                               runing_time, 1.2)
                operation_finished = False

            if (operation_finished == False):    
                operation_finished = PS.transition_to_pose(cmd, runing_time)




        #stand down
        if (runing_time > 5.0 and runing_time <= 5.0 + 2.5):
            if(operation_finished == True):
                PS.prepare_for_new_transition(motors_readout, stand_down_joint_pos,
                               runing_time, 2.5)
                operation_finished = False
                
            if (operation_finished == False):    
                operation_finished = PS.transition_to_pose(cmd, runing_time)






        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)



