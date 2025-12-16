
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

from GO2_IK import Go2LegIK


from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import sys, time

# Neutral foot positions in hip frame (x forward, y left, z down)
NEUTRAL_FOOT = {
    'FR': np.array([ 0.08, -0.09, -0.30]),
    'FL': np.array([ 0.08,  0.09, -0.30]),
    'RR': np.array([-0.08, -0.09, -0.30]),
    'RL': np.array([-0.08,  0.09, -0.30]),
}

# Stand pose joint triplet guessed from your earlier file (hip_abd, hip_pitch, calf)
# Keep these as the safe fallback
STAND_UP_POS = np.array([
    0.00571868, 0.608813, -1.21763,
   -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763,
   -0.00571868, 0.608813, -1.21763
])



DT          = 0.002
KP, KD      = 150.0, 2.0


LEGS          = ['FR','FL','RR','RL']



# ---------------------- state helpers ----------------------
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

# ---------------------- main loop (static values) ----------------------
def main():
    # init comms (static LO)
    #init_channel_with_fallback('lo', domains=range(0,6))
    ChannelFactoryInitialize(1, 'lo')


    pub = ChannelPublisher('rt/lowcmd', LowCmd_)
    pub.Init()
    sub = ChannelSubscriber('rt/lowstate', LowState_)
    sub.Init()


    ik = Go2LegIK(l_thigh=0.213, l_calf=0.213, hip_abd_sign=1.0, calf_sign=-1.0)

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
    printed_debug = False

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


            phase = 0.5 * np.pi * np.sin(2.0 * np.pi * 0.5 * t)  # 0.5 Hz up/down
            dz = 0.04 * (phase)  # +/- 5 cm

            # default pose
            res_full = ik.full_leg_ik(0, 0.0, 0.25)

            # per-leg constant joint values (use stand pose)
            for idx, leg in enumerate(LEGS):
                hi, ti, ci = leg_indices(leg)
                # Use the stand pose joint angles for each leg
                cmd.motor_cmd[hi].q = res_full.hip_abd   # prawo lewo 
                cmd.motor_cmd[ti].q = -res_full.hip_pitch - np.pi/2 # przod tyl
                cmd.motor_cmd[ci].q = res_full.knee   # kolano

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
        print("Interrupted by user — stopping.")

    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)
    print("Done, exiting.")

if __name__ == '__main__':
    main()



