"""
Error-state tracking MPC for the quadrotor.
"""

import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import osqp
import scipy.sparse as sp
from control import dlqr
import mujoco as mj
from quad_model import (
    NU,
    NX,
    U_MAX,
    U_MIN,
    A_stack_vmap,
    B_stack_vmap,
    E_pinv_stack_vmap,
    E_stack_vmap,
    describe,
    h,
    mj_to_x,
    model,
    new_data,
    quad_dynamics_rk4,
    x_to_mj,
)
from mujoco.glfw import glfw
from renderer import MujocoRenderer
from trajectory_generator import TRAJECTORIES, TrajectoryGenerator

_rk4_stack = jax.jit(jax.vmap(quad_dynamics_rk4, in_axes=(0, 0)))


@jax.jit
def _linearize(x_refs, u_refs):
    """
    Everything the QP needs from the reference, in one device call.
    """
    x_lin, u_lin = x_refs[:-1], u_refs[:-1]
    A = A_stack_vmap(x_lin, u_lin)
    B = B_stack_vmap(x_lin, u_lin)
    E = E_stack_vmap(x_refs[:, 3:7])
    Ep = E_pinv_stack_vmap(x_refs[:, 3:7])
    defect = _rk4_stack(x_lin, u_lin) - x_refs[1:]
    return (
        Ep[1:] @ A @ E[:-1],
        Ep[1:] @ B,
        jnp.einsum("kij,kj->ki", Ep[1:], defect),
        Ep[0],
    )


Q_DIAG = np.array([120, 120, 120, 8, 8, 4, 2, 2, 2, 0.5, 0.5, 0.5], float)
R_DIAG = np.full(NU, 0.05)


class TrackingMPC:
    def __init__(
        self,
        traj,
        Nh=20,
        q_diag=Q_DIAG,
        r_diag=R_DIAG,
        ctrl_dt=h,
        terminal_every=25,
        box=True,
        polish=False,
    ):
        self.traj = traj
        self.Nh = Nh
        self.box = box
        self.ctrl_dt = ctrl_dt
        self.terminal_every = terminal_every
        self.Q = np.diag(np.asarray(q_diag, float))
        self.R = np.diag(np.asarray(r_diag, float))

        self._build_constraint_template()
        self._build_hessian()

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=self.H,
            q=np.zeros(self.n_var),
            A=self.A_template,
            l=self.l,
            u=self.u,
            verbose=False,
            warm_start=True,
            polish=polish,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=4000,
        )

        self.P_term = None
        self.reset()

    def reset(self):
        """
        Forget the previous run.  Mandatory after mj_resetData.
        """
        self.u_last = np.zeros(NU)
        self.t_next = -np.inf
        self.stats = dict(solves=0, fails=0, t_ms=[], iters=[], e_pos=[], e_att=[])
        self.x_ref0 = None
        self.dx0 = np.zeros(NX)
        self.k_tick = 0  # also forces the terminal cost to refresh on tick 0

    def _build_constraint_template(self):
        """
        fill every varying entry with (flat index + 1) so it can be
        located in the CSC data array afterwards.  The +1 is because a literal 0
        marker gets dropped as a structural zero.
        """
        Nh = self.Nh
        n_var = Nh * (NU + NX)
        counter = 0
        B_mark, A_mark = [], []
        for _ in range(Nh):  # B_0 .. B_{Nh-1}
            blk = np.arange(counter, counter + NX * NU) + 1.0
            B_mark.append(sp.csc_matrix(blk.reshape(NX, NU)))
            counter += NX * NU
        for _ in range(Nh - 1):  # A_1 .. A_{Nh-1}
            blk = np.arange(counter, counter + NX * NX) + 1.0
            A_mark.append(sp.csc_matrix(blk.reshape(NX, NX)))
            counter += NX * NX
        self.n_flat = counter

        I_neg = -sp.identity(NX, format="csc")
        Zu, Zx = sp.csc_matrix((NX, NU)), sp.csc_matrix((NX, NX))
        rows = [[B_mark[0], I_neg] + [Zu, Zx] * (Nh - 1)]
        for k in range(1, Nh):
            rows.append(
                [Zu, Zx] * (k - 1)
                + [Zu, A_mark[k - 1], B_mark[k], I_neg]
                + [Zu, Zx] * (Nh - 1 - k)
            )
        blocks = rows

        if self.box:
            sel = -2.0 * sp.identity(NU, format="csc")
            Zux = sp.csc_matrix((NU, NX))
            for k in range(Nh):
                blocks.append(
                    [Zux if j % 2 else sp.csc_matrix((NU, NU)) for j in range(2 * Nh)]
                )
                blocks[-1][2 * k] = sel

        A_template = sp.bmat(blocks).tocsc()
        d = A_template.data
        var = np.where(d > 0.0)[0]
        self.var_loc = var
        self.flat_map = d[var].astype(np.int64) - 1
        self.A_data = np.zeros_like(d)
        self.A_data[np.where(d == -1.0)[0]] = -1.0
        self.A_data[np.where(d == -2.0)[0]] = 1.0
        A_template.data = self.A_data.copy()

        self.A_template = A_template
        self.n_var = n_var
        self.n_dyn = Nh * NX
        self.n_con = A_template.shape[0]
        self.flat = np.zeros(self.n_flat)
        self.l = np.zeros(self.n_con)
        self.u = np.zeros(self.n_con)

    def _build_hessian(self):
        Nh = self.Nh
        stage = sp.bmat([[self.R, None], [None, self.Q]], format="csc")
        H_stage = sp.kron(sp.eye(Nh - 1, format="csc"), stage, format="csc")
        # Placeholder for the terminal block.  Only its pattern matters
        P_holder = sp.triu(
            sp.csc_matrix(np.ones((NX, NX)) + NX * np.eye(NX)), format="csc"
        )
        H_term = sp.bmat([[self.R, None], [None, P_holder]], format="csc")
        H = sp.block_diag([H_stage, H_term], format="csc")
        col0 = (Nh - 1) * (NU + NX) + NU
        n_P = (NX * (NX + 1)) // 2
        self.Px_idx = np.arange(H.indptr[col0], H.indptr[col0] + n_P)
        self.H = H

    def __call__(self, _model, data):
        """MuJoCo control callback."""
        if data.time < self.t_next:
            data.ctrl[:] = self.u_last  # hold between control instants
            return
        self.t_next = data.time + self.ctrl_dt - 1e-12
        t0 = time.perf_counter()

        x_cur = mj_to_x(data)
        x_refs, u_refs = self.traj.get_refs_batched(
            data.time, h, self.Nh, q_anchor=x_cur[3:7]
        )
        A_t, B_t, d, Ep0 = jax.device_get(_linearize(x_refs, u_refs))
        u_ref = np.asarray(u_refs)
        x_ref0 = np.asarray(x_refs[0])

        dx0 = Ep0 @ (x_cur - x_ref0)
        self.x_ref0, self.dx0 = x_ref0, dx0
        self.stats["e_pos"].append(float(np.linalg.norm(dx0[0:3])))
        self.stats["e_att"].append(float(np.linalg.norm(dx0[3:6])))

        # terminal cost: LQR of the last linearization, refreshed occasionally
        if self.k_tick % self.terminal_every == 0 or self.P_term is None:
            try:
                _, P, _ = dlqr(A_t[-1], B_t[-1], self.Q, self.R)
                self.P_term = np.asarray(P)
                self.Px_new = sp.triu(sp.csc_matrix(self.P_term), format="csc").data
            except Exception as exc:  # unstabilizable linearization
                if self.P_term is None:
                    raise
                print(f"[mpc] dlqr failed at t={data.time:.2f}: {exc}")
                self.Px_new = None
        else:
            self.Px_new = None

        # constraint bounds
        rhs = -d
        rhs[0] -= A_t[0] @ dx0
        self.l[: self.n_dyn] = rhs.ravel()
        self.u[: self.n_dyn] = self.l[: self.n_dyn]
        if self.box:
            lo = (U_MIN[None, :] - u_ref[:-1]).ravel()
            hi = (U_MAX[None, :] - u_ref[:-1]).ravel()
            self.l[self.n_dyn :] = lo
            self.u[self.n_dyn :] = hi

        # constraint matrix
        nB = self.Nh * NX * NU
        self.flat[:nB] = B_t.reshape(-1)
        self.flat[nB:] = A_t[1:].reshape(-1)
        self.A_data[self.var_loc] = self.flat[self.flat_map]

        if self.Px_new is None:
            self.prob.update(Ax=self.A_data, l=self.l, u=self.u)
        else:
            self.prob.update(
                Px=self.Px_new, Px_idx=self.Px_idx, Ax=self.A_data, l=self.l, u=self.u
            )
        res = self.prob.solve()

        if res.info.status_val in (1, 2):
            u_cmd = u_ref[0] + res.x[:NU]
        else:
            self.stats["fails"] += 1
            print(
                f"[mpc] t={data.time:6.2f}  QP {res.info.status} -> feed-forward only"
            )
            u_cmd = u_ref[0]

        self.u_last = np.clip(u_cmd, U_MIN, U_MAX)
        data.ctrl[:] = self.u_last
        self.stats["solves"] += 1
        self.stats["iters"].append(int(res.info.iter))
        self.stats["t_ms"].append(1e3 * (time.perf_counter() - t0))
        self.k_tick += 1

    def tracking(self):
        ep = np.asarray(self.stats["e_pos"] or [0.0])
        ea = np.degrees(np.asarray(self.stats["e_att"] or [0.0]))
        return (
            f"tracking: pos rms {np.sqrt((ep**2).mean()) * 100:5.1f} cm, "
            f"max {ep.max() * 100:5.1f} cm | att rms {np.sqrt((ea**2).mean()):4.1f} deg, "
            f"max {ea.max():4.1f} deg"
        )

    def timing(self):
        t = (
            np.asarray(self.stats["t_ms"][1:])
            if len(self.stats["t_ms"]) > 1
            else np.zeros(1)
        )
        it = (
            np.asarray(self.stats["iters"][1:])
            if len(self.stats["iters"]) > 1
            else np.zeros(1)
        )
        return (
            f"mpc: {self.stats['solves']} solves, {self.stats['fails']} failed | "
            f"{t.mean():.2f} ms mean, {np.percentile(t, 95):.2f} ms p95, "
            f"{t.max():.2f} ms max | {it.mean():.0f} osqp iters "
            f"(budget {1e3 * self.ctrl_dt:.0f} ms)"
        )


# Demo
KEY_HELP = (
    "space pause   r reset   [ ] speed   tab next trajectory   "
    "c clear trail   f follow cam"
)


def main():
    names = [n for n in TRAJECTORIES if n != "setpoint"]
    ap = argparse.ArgumentParser(
        description=__doc__.split("Run it:")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--traj", default="figure8", choices=names)
    ap.add_argument("--speed", type=float, default=1.0, help="time-warp factor")
    ap.add_argument("--yaw", default=None, help="fixed | spin | follow | center")
    ap.add_argument("--Nh", type=int, default=20, help="horizon length")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until closed")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    traj = TrajectoryGenerator(args.traj, speed=args.speed, yaw_mode=args.yaw)
    ctrl = TrackingMPC(traj, Nh=args.Nh)
    data = new_data()
    print(describe())
    print(traj.report())

    x0 = np.asarray(traj.get_refs_batched(0.0, h, 1)[0][0])
    x_to_mj(data, x0)
    mj.mj_forward(model, data)

    if args.headless:
        n = int((args.duration or 10.0) / h)
        for _ in range(n):
            ctrl(model, data)
            mj.mj_step(model, data)
        print(ctrl.tracking())
        print(ctrl.timing())
        return

    view = MujocoRenderer(
        model, data, title=f"tracking MPC - {args.traj}", follow=True, follow_body="x2"
    )
    view.set_path("ref", traj.path_points(400))
    mj.set_mjcb_control(ctrl)

    def launch(traj):
        """
        Put it on a reference before handing it over.
        """
        traj.reset()
        ctrl.reset()
        x_to_mj(data, np.asarray(traj.get_refs_batched(data.time, h, 1)[0][0]))
        mj.mj_forward(model, data)
        view.set_path("ref", traj.path_points(400))
        view.clear_trail()

    paused = False
    idx = names.index(args.traj)
    print(KEY_HELP)

    try:
        while not view.is_window_closed():
            for key in view.take_pressed():
                if key == glfw.KEY_SPACE:
                    paused = not paused
                elif key == glfw.KEY_R:
                    mj.mj_resetData(model, data)
                    launch(traj)
                elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
                    step = 1.25 if key == glfw.KEY_RIGHT_BRACKET else 1 / 1.25
                    traj.set_speed(traj.speed * step)
                    view.set_path("ref", traj.path_points(400))
                    print(traj.report())
                elif key == glfw.KEY_TAB:
                    idx = (idx + 1) % len(names)
                    traj = TrajectoryGenerator(names[idx], speed=traj.speed)
                    ctrl.traj = traj
                    launch(traj)
                    print(traj.report())
                elif key == glfw.KEY_C:
                    view.clear_trail()
                elif key == glfw.KEY_F:
                    view.follow = not view.follow

            if not paused:
                t_frame = data.time
                while data.time - t_frame < 1.0 / 60.0:
                    mj.mj_step(model, data)
                view.push_trail(data.xipos[model.body("x2").id])

            e = ctrl.dx0
            view.set_hud(
                f"{traj.name}  speed x{traj.speed:.2f}{'  [PAUSED]' if paused else ''}\n"
                f"t {data.time:6.2f} s\n"
                f"pos err {np.linalg.norm(e[0:3]) * 100:5.1f} cm\n"
                f"att err {np.degrees(np.linalg.norm(e[3:6])):5.1f} deg\n"
                f"solve   {ctrl.stats['t_ms'][-1] if ctrl.stats['t_ms'] else 0:5.2f} ms",
                KEY_HELP.replace("   ", "\n"),
            )
            view.render()
            if args.duration and data.time >= args.duration:
                break
    finally:
        mj.set_mjcb_control(None)
        view.close()
        print(ctrl.tracking())
        print(ctrl.timing())


if __name__ == "__main__":
    main()
