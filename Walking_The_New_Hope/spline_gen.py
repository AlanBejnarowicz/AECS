import time
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

PHASE_OFF = {'FR':0.0, 'RL':0.0, 'FL':0.5, 'RR':0.5}
#PHASE_OFF = {'FR':0.0, 'RL':0.5, 'FL':0.5, 'RR':0.0}

NEUTRAL_FOOT = {
    'FR': np.array([ 0.08, -0.09, -0.35]),
    'FL': np.array([ 0.08,  0.09, -0.35]),
    'RR': np.array([-0.08, -0.09, -0.35]),
    'RL': np.array([-0.08,  0.09, -0.35]),
}

# ---------------------- SplineGenerator class ----------------------
class SplineGenerator:
    """
    Generates cubic-spline swing trajectories for a foot.
    Usage:
      sg = SplineGenerator()
      spline = sg.plan_swing(start_pos, end_pos, lift_height)
      pos = spline(0.0 .. 1.0)  # returns (x,y,z)
      sg.plot()  # plots last planned spline
    """
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        self._last_cs = None  # store CubicSpline objects for x,y,z
        self._last_t = None
        self._last_pts = None

    def plan_swing(self, p0, p1, lift=0.04, inverse=False):
        """
        Plan a swing from p0 to p1 (3d vectors), returning a callable f(s) where s in [0,1].
        The path is a 1D parametric spline over s (normalized time).
        We build splines for x,y,z separately with a mid-point lift for z.
        If inverse=True, the trajectory goes from p1 to p0.
        """
        p0 = np.asarray(p0).reshape(3)
        p1 = np.asarray(p1).reshape(3)

        if inverse:
            p0, p1 = p1, p0

        # mid point: halfway in x/y, with lifted z
        pm = 0.5*(p0 + p1)
        pm[2] = min(p0[2], p1[2]) + lift  # negative z = down; subtract to lift up

        # knot positions in normalized time
        t = np.array([0.0, 0.3, 0.6, 1.0])
        # Add midpoint for cubic spline control points
        x_pts = np.array([p0[0], p0[0], p1[0], p1[0]])
        y_pts = np.array([p0[1], pm[1], pm[1], p1[1]])
        z_pts = np.array([p0[2], pm[2], pm[2], p1[2]])

        cs_x = CubicSpline(t, x_pts, bc_type='clamped')
        cs_y = CubicSpline(t, y_pts, bc_type='clamped')
        cs_z = CubicSpline(t, z_pts, bc_type='clamped')

        self._last_cs = (cs_x, cs_y, cs_z)
        self._last_t = t
        self._last_pts = np.stack([x_pts, y_pts, z_pts], axis=1)

        def spline(s):
            s = np.clip(s, 0.0, 1.0)
            return np.stack([cs_x(s), cs_y(s), cs_z(s)], axis=-1)

        return spline

    @property
    def plot(self):
        """
        Returns a function that plots the last planned spline when called.
        Example: sg.plot()  # shows a 3D plot of the foot path
        """
        def _do_plot(samples=200):
            if self._last_cs is None:
                raise RuntimeError("No spline planned yet (call plan_swing first).")
            cs_x, cs_y, cs_z = self._last_cs
            s = np.linspace(0.0, 1.0, samples)
            pts = np.vstack([cs_x(s), cs_y(s), cs_z(s)]).T

            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(pts[:,0], pts[:,1], pts[:,2], label='swing path')
            ax.scatter(self._last_pts[:,0], self._last_pts[:,1], self._last_pts[:,2], c='r', label='control pts')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
            ax.set_title('Foot swing spline')
            ax.legend()
            plt.show()
        return _do_plot
    
# ---------------------- SplineGaitController (unchanged) ----------------------
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
        half = self.step_length / 2.0
        self.stance_start_offset = np.array([ half * self.forward_dir, 0.0, 0.0])
        self.stance_end_offset   = np.array([-half * self.forward_dir, 0.0, 0.0])

    def phase_of_time(self, t, offset):
        cycle_pos = ((t / self.cycle_time) + offset) % 1.0
        return cycle_pos

    def foot_target(self, leg, t):
        # +1 for front legs, -1 for rear legs
        # (if your robot uses a different convention, flip the mapping below)
        x_sign_map = {'FR': +1.0, 'FL': +1.0, 'RR': -1.0, 'RL': -1.0}
        sx = x_sign_map.get(leg, 1.0)

        offset = PHASE_OFF[leg]
        phase = self.phase_of_time(t, offset)

        if phase < self.duty:
            # --- STANCE: move linearly from +half to -half along X (apply per-leg sign) ---
            s = phase / self.duty
            delta = (1.0 - s) * self.stance_start_offset + s * self.stance_end_offset
            delta = delta.copy()

            #delta[0] *= sx            # flip X motion for rear legs only
            if (leg == 'FR') or (leg == 'FL'):
                delta[0] *= self.forward_dir  # apply forward direction for front legs

            target = self.neutral[leg] + delta
            return target
        else:
            # --- SWING: plan spline from stance_end -> stance_start with same per-leg sign ---
            s = (phase - self.duty) / (1.0 - self.duty)

            p_start = self.neutral[leg] + self.stance_end_offset * sx
            p_end   = self.neutral[leg] + self.stance_start_offset * sx

            if (leg == 'FR') or (leg == 'FL'):
                spline = self.sg.plan_swing(p_start, p_end, lift=self.step_height)

            if (leg == 'RR') or (leg == 'RL'):
                spline = self.sg.plan_swing(p_start, p_end, lift=self.step_height, inverse=True)


            return spline(s)


    

if __name__ == "__main__":
    # gait and spline setup
    spline_gen = SplineGenerator()
    gait = SplineGaitController(NEUTRAL_FOOT, spline_gen,
                            step_length=0.18, step_height=0.1,
                            cycle_time=0.6, duty=0.6,
                            forward_dir=1.0)  # flip walking direction
    
    # plot foot trajectory for one leg
    spline_gen.plan_swing(NEUTRAL_FOOT['FR'] + gait.stance_end_offset,
                          NEUTRAL_FOOT['FR'] + gait.stance_start_offset,
                          lift=gait.step_height)
    spline_gen.plot()

