"""
Fly the X2 by hand: the keyboard drives a velocity setpoint, the setpoint is
turned into a dynamically-consistent reference by the differential-flatness
generator.
"""

import argparse

import numpy as np

import mujoco as mj
from mujoco.glfw import glfw
from quad_model import describe, g, h, model, new_data, x_to_mj
from renderer import MujocoRenderer
from disturbances import WEIGHT, add_disturbance_arguments, disturbance_from_args
from flight_visuals import DisplayOptions, add_display_arguments
from tracking_mpc_optimized import TrackingMPC
from trajectory_generator import TrajectoryGenerator

HOME = np.array([0.0, 0.0, 2.0])
Z_FLOOR = 0.35
TILT_MAX = np.radians(22.0)  # keeps the reference inside the rotors' reach


class Pilot:
    """
    Turns key state into the live parameters of the setpoint trajectory.
    The velocity command is passed through a critically damped 2nd-order filter
    so that position, velocity and acceleration are continuous. the MPC
    linearizes about this reference, and a step in commanded acceleration would
    mean a step in commanded tilt.  The filter is written in terms of a time
    constant, not a per-frame blend factor, so the feel does not change with
    the frame rate.
    """

    def __init__(self, traj, speed=1.5, yaw_speed=1.2, tau=0.35):
        self.traj = traj
        self.speed = speed
        self.yaw_speed = yaw_speed
        self.tau = tau
        self.auto_yaw = False
        self.w = 2.0 / max(tau, 1e-3)  # critically damped, both poles at -w
        self.dv_max = g * np.tan(TILT_MAX) * np.e / self.w
        self.r = HOME.copy()
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.cmd = np.zeros(3)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.frame_yaw = 0.0
        self.goal = None
        self._push()

    def _push(self):
        """
        Hand the MPC the filter's state and target.
        the horizon is then the exact future of this filter rather than a frozen quadratic.
        """
        self.traj.live = {
            "r": self.r.copy(),
            "v": self.v.copy(),
            "a": self.a.copy(),
            "cmd": self.cmd.copy(),
            "w": float(self.w),
            "yaw": float(self.yaw),
            "yaw_rate": float(self.yaw_rate),
        }

    def go_home(self):
        """
        Ask to be flown home.
        """
        self.goal = HOME.copy()
        self.auto_yaw = False

    def teleport(self, r=HOME):
        self.goal = None
        self.r, self.v, self.a = np.array(r, float), np.zeros(3), np.zeros(3)
        self.cmd = np.zeros(3)
        self.yaw = self.yaw_rate = self.frame_yaw = 0.0
        self._push()

    def scale_speed(self, factor):
        self.speed = float(np.clip(self.speed * factor, 0.2, 6.0))
        return self.speed

    def update(self, keys, dt):
        boost = (
            2.0 if (keys[glfw.KEY_LEFT_SHIFT] or keys[glfw.KEY_RIGHT_SHIFT]) else 1.0
        )
        sp = self.speed * boost
        fwd = float(keys[glfw.KEY_W]) - float(keys[glfw.KEY_S])
        left = float(keys[glfw.KEY_A]) - float(keys[glfw.KEY_D])
        up = float(keys[glfw.KEY_UP]) - float(keys[glfw.KEY_DOWN])
        yaw_in = float(keys[glfw.KEY_LEFT]) - float(keys[glfw.KEY_RIGHT])
        yaw_in += float(keys[glfw.KEY_Q]) - float(keys[glfw.KEY_E])
        c, s = np.cos(self.frame_yaw), np.sin(self.frame_yaw)
        cmd = np.array(
            [-sp * (fwd * c - left * s), -sp * (fwd * s + left * c), sp * up]
        )

        if self.goal is not None:
            if fwd or left or up:  # any stick input takes control straight back
                self.goal = None
            else:
                # Proportional approach, saturated at cruise speed and eased
                # inside a 1 m ball so the arrival has no velocity step.
                d = self.goal - self.r
                n = float(np.linalg.norm(d))
                cmd = d * (min(sp, 1.5 * n) / n) if n > 1e-3 else np.zeros(3)
                if n < 0.02:
                    self.goal = None

        dv = cmd - self.v
        n = float(np.linalg.norm(dv))
        if n > self.dv_max:
            cmd = self.v + dv * (self.dv_max / n)

        # critically damped 2nd order on velocity
        w = self.w
        self.a = self.a + dt * (w * w * (cmd - self.v) - 2.0 * w * self.a)
        self.v = self.v + self.a * dt
        self.r = self.r + self.v * dt

        if self.auto_yaw and np.linalg.norm(self.v[:2]) > 0.25 and not yaw_in:
            want = np.arctan2(self.v[1], self.v[0]) - np.pi
            err = (want - self.yaw + np.pi) % (2.0 * np.pi) - np.pi
            self.yaw_rate = np.clip(2.0 * err, -self.yaw_speed, self.yaw_speed)
        else:
            self.yaw_rate = yaw_in * self.yaw_speed * boost
        if not self.auto_yaw:
            self.frame_yaw = self.yaw  # nose-relative: re-latched every tick
        self.yaw += self.yaw_rate * dt  # never wrapped: cos/sin do not care

        if self.r[2] < Z_FLOOR:  # do not command the drone into the ground
            self.r[2] = Z_FLOOR
            self.v[2] = max(self.v[2], 0.0)
            self.a[2] = max(self.a[2], 0.0)
            cmd[2] = max(cmd[2], 0.0)  # keep the horizon out of the floor too
        self.cmd = cmd
        self._push()


def main(args=None, dist=None, frame_callback=None):
    if args is None:
        ap = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        add_disturbance_arguments(ap)
        add_display_arguments(ap)
        args = ap.parse_args()
    body = model.body("x2").id
    if dist is None:
        dist = disturbance_from_args(args, body)
    traj = TrajectoryGenerator("setpoint")
    pilot = Pilot(traj)
    ctrl = TrackingMPC(traj)
    data = new_data()

    x_to_mj(data, np.concatenate([HOME, [1, 0, 0, 0], np.zeros(6)]))
    mj.mj_forward(model, data)

    print(describe())
    print(__doc__.split("Controls")[1])
    print(
        f"push strength: {args.push:.2f} x weight ({args.push * WEIGHT:.2f} N per axis)"
    )
    if dist.armed:
        print(dist.plan())

    view = MujocoRenderer(
        model,
        data,
        title="X2 - keyboard flight",
        width=getattr(args, "width", 1200),
        height=getattr(args, "height", 900),
        follow=getattr(args, "follow", True),
        follow_body="x2",
        display=DisplayOptions.from_args(args),
        frame_callback=frame_callback,
    )
    mj.set_mjcb_control(ctrl)
    paused = False
    dt_frame = 1.0 / 60.0

    try:
        while not view.is_window_closed():
            for key in view.take_pressed():
                if view.display.handle_key(key):
                    continue
                elif key == glfw.KEY_SPACE:
                    paused = not paused
                elif key == glfw.KEY_H:
                    pilot.go_home()
                elif key == glfw.KEY_G:
                    pilot.auto_yaw = not pilot.auto_yaw
                    pilot.frame_yaw = pilot.yaw  # course lock starts at the nose
                elif key == glfw.KEY_RIGHT_BRACKET:
                    pilot.scale_speed(1.25)
                elif key == glfw.KEY_LEFT_BRACKET:
                    pilot.scale_speed(1 / 1.25)
                elif key == glfw.KEY_0:
                    pilot.speed = 1.5
                elif key == glfw.KEY_C:
                    view.clear_trail()
                elif key == glfw.KEY_F:
                    view.follow = not view.follow
                elif key == glfw.KEY_R:
                    mj.mj_resetData(model, data)
                    x_to_mj(data, np.concatenate([HOME, [1, 0, 0, 0], np.zeros(6)]))
                    mj.mj_forward(model, data)
                    pilot.teleport()  # reference and vehicle move together
                    ctrl.reset()  # ... and the control clock rewinds too
                    dist.reset()
                    view.arrows.clear()
                    view.clear_trail()

            if not paused:
                t_frame = data.time
                while data.time - t_frame < dt_frame:
                    pilot.update(view.keys, h)
                    dist.set_hold(view.keys, args.push)
                    dist.step(data)
                    mj.mj_step(model, data)
                view.push_trail(data.xipos[body])
                view.set_path(
                    "cmd",
                    np.stack([pilot.r, data.xipos[body]]),
                    rgba=(1.0, 0.3, 0.3, 0.8),
                    width=0.006,
                )

            view.set_arrow(
                "push",
                data.xipos[body],
                dist.f,
                metres_per_newton=args.disturb_scale / WEIGHT,
            )
            e = ctrl.dx0
            boost = view.keys[glfw.KEY_LEFT_SHIFT] or view.keys[glfw.KEY_RIGHT_SHIFT]
            view.set_hud(
                f"speed {pilot.speed:4.2f} m/s{'  TURBO' if boost else ''}"
                f"{'  [PAUSED]' if paused else ''}\n"
                f"pos  {data.qpos[0]:6.2f} {data.qpos[1]:6.2f} {data.qpos[2]:6.2f}\n"
                f"|v|  {np.linalg.norm(data.qvel[0:3]):5.2f} m/s\n"
                f"yaw  {np.degrees(pilot.yaw) % 360:5.1f} deg"
                f"{'  (auto)' if pilot.auto_yaw else ''}\n"
                f"err  {np.linalg.norm(e[0:3]) * 100:5.1f} cm\n"
                f"solve {ctrl.stats['t_ms'][-1] if ctrl.stats['t_ms'] else 0:5.2f} ms\n"
                f"{dist.hud()}",
                "WASD move\narrows alt/yaw\nshift turbo\n[ ] speed\n"
                "g auto-yaw\nh home\nr reset\nspace pause\n"
                "I/K J/L U/O push\nF1 guide  F2 keys\nF3 trail  F4 clean\n"
                "F5 vectors  F6 paths",
            )
            view.render()
            if getattr(args, "duration", 0.0) > 0 and data.time >= args.duration:
                break
    finally:
        mj.set_mjcb_control(None)
        view.close()
        print(ctrl.tracking())
        print(ctrl.timing())


if __name__ == "__main__":
    main()
