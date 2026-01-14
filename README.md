# Quadruped walking algorithms Spline Approach for Trajectory Generation
## Adaptive embodied control systems

This is a repo containing a spline trajectory generator and IK for Unitree GO2 robot,
simulated in Mujooco Sim. 

<img width="2880" height="1800" alt="mujacoo_sim" src="https://github.com/user-attachments/assets/27653c3d-136c-4d54-b010-c575245f5e32" />

### Executing program
1. Source venv
```
source mujoco_venv/bin/activate
```

2. Start Mujoco Unitree [First install it from Unitree Repo]
```
cd unitree_mujoco-main/
./simulate/build/unitree_mujoco 
```

3. Start leg trajectory generator algorithm [In VENV from step 1]
```
python3 Walking_The_New_Hope/spline_walk_PID.py
```
Robot should start walking

### Tuning parameters
In python script spline_walk_PID.py starting from line 251 are variables responsible for generating
trajectory and balancing. By tuning these parameters better walking resoults could be achieved.

Most important ones are step_length=0.18, step_height=0.1, cycle_time=0.3, duty=0.6 beeing responsible for 
leg speed and step height. Another modification to that could be done in SplineGaitController class to tune even more 
spline points.
```
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
```

<img width="1561" height="1145" alt="robot_in_terrain" src="https://github.com/user-attachments/assets/f2f4e134-8292-4350-873e-525c8d8b8a4c" />
