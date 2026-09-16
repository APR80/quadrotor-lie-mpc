"""
Differential-flatness trajectory library.

A trajectory is defined by ONE function of time
plus a heading reference, everything else is derived so adding a trajectory
means writing its position curve and nothing else
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from quad_model import (
    MIXER_INV,
    J,
    NU,
    NX_FULL,
    g,
    m,
    quat_align,
    vee_jnp,
)

_EPS = 1e-9
_E3 = jnp.array([0.0, 0.0, 1.0])
_J = jnp.asarray(J)
_MIXER_INV = jnp.asarray(MIXER_INV)


@jit
def _unit(v):
    """Normalize without the kink at |v| = 0 (keeps autodiff well behaved)."""
    return v / jnp.sqrt(jnp.sum(v * v) + _EPS**2)


@jit
def rot_to_quat(R):
    trace = jnp.trace(R)

    def positive_trace():
        r = jnp.sqrt(1.0 + trace)
        ri = 0.5 / r
        return jnp.array(
            [
                0.5 * r,
                ri * (R[2, 1] - R[1, 2]),
                ri * (R[0, 2] - R[2, 0]),
                ri * (R[1, 0] - R[0, 1]),
            ]
        )

    def largest_diagonal():
        def x_major():
            r = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            ri = 0.5 / r
            return jnp.array(
                [
                    ri * (R[2, 1] - R[1, 2]),
                    0.5 * r,
                    ri * (R[0, 1] + R[1, 0]),
                    ri * (R[0, 2] + R[2, 0]),
                ]
            )

        def y_major():
            r = jnp.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
            ri = 0.5 / r
            return jnp.array(
                [
                    ri * (R[0, 2] - R[2, 0]),
                    ri * (R[0, 1] + R[1, 0]),
                    0.5 * r,
                    ri * (R[1, 2] + R[2, 1]),
                ]
            )

        def z_major():
            r = jnp.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
            ri = 0.5 / r
            return jnp.array(
                [
                    ri * (R[1, 0] - R[0, 1]),
                    ri * (R[0, 2] + R[2, 0]),
                    ri * (R[1, 2] + R[2, 1]),
                    0.5 * r,
                ]
            )

        i = jnp.argmax(jnp.array([R[0, 0], R[1, 1], R[2, 2]]))
        return jax.lax.switch(i, [x_major, y_major, z_major])

    q = jax.lax.cond(trace > 0.0, positive_trace, largest_diagonal)
    return q / (jnp.linalg.norm(q) + _EPS)


# Position curves.
def hover(t, p, center=(0.0, 0.0, 2.0)):
    return jnp.asarray(center) + 0.0 * t


def circle(t, p, R=2.0, f=1.0, zd=2.0):
    return jnp.array([R * jnp.cos(f * t), R * jnp.sin(f * t), zd])


def figure8(t, p, Rx=3.0, Ry=2.0, f=0.8, zd=2.0):
    """Lemniscate of Gerono: one x lap per two y laps."""
    return jnp.array([Rx * jnp.sin(f * t), Ry * jnp.sin(2 * f * t), zd])


def lissajous(
    t, p, A=(2.5, 2.0, 0.6), n=(2.0, 3.0, 2.0), phase=(0.0, 0.0, 0.0), f=0.6, zd=2.2
):
    """3D Lissajous figure."""
    A, n, phase = jnp.asarray(A), jnp.asarray(n), jnp.asarray(phase)
    xy = A[:2] * jnp.sin(n[:2] * f * t + phase[:2])
    z = zd + A[2] * jnp.sin(n[2] * f * t + phase[2])
    return jnp.array([xy[0], xy[1], z])


def helix(t, p, R=2.0, f=0.9, zd=2.0, dz=1.2, fz=0.3):
    return jnp.array(
        [R * jnp.cos(f * t), R * jnp.sin(f * t), zd + dz * jnp.sin(fz * t)]
    )


def rose(t, p, A=2.4, k=5.0, f=0.5, zd=2.0):
    th = f * t
    rad = A * jnp.cos(k * th)
    return jnp.array([rad * jnp.cos(th), rad * jnp.sin(th), zd])


def torus_knot(t, p, R=2.0, r=0.7, pw=2.0, qw=3.0, f=0.45, zd=2.2):
    a, b = pw * f * t, qw * f * t
    rad = R + r * jnp.cos(b)
    return jnp.array([rad * jnp.cos(a), rad * jnp.sin(a), zd + r * jnp.sin(b)])


def epitrochoid(t, p, R=1.6, r=0.45, d=0.8, f=0.55, zd=2.0):
    """Spirograph curve traced by a circle rolling outside another circle."""
    k = (R + r) / r
    return jnp.array(
        [
            (R + r) * jnp.cos(f * t) - d * jnp.cos(k * f * t),
            (R + r) * jnp.sin(f * t) - d * jnp.sin(k * f * t),
            zd,
        ]
    )


def heart(t, p, A=0.11, f=0.5, zd=2.5, y=0.0):
    s = f * t
    hx = 16.0 * jnp.sin(s) ** 3
    hz = (
        13.0 * jnp.cos(s) - 5.0 * jnp.cos(2 * s) - 2.0 * jnp.cos(3 * s) - jnp.cos(4 * s)
    )
    return jnp.array([A * hx, y, zd + A * (hz + 6.0)])


def squircle(t, p, A=2.2, power=4.0, f=0.6, zd=2.0):
    """
    Rounded so all four derivatives stay finite.
    """
    c, s = jnp.cos(f * t), jnp.sin(f * t)
    scale = (c**power + s**power) ** (1.0 / power)
    return jnp.array([A * c / scale, A * s / scale, zd])


def setpoint(t, p):
    w = p["w"]
    e0 = p["v"] - p["cmd"]
    b = p["a"] + w * e0
    E = jnp.exp(-w * t)
    return (
        p["r"]
        + p["cmd"] * t
        + e0 * (1.0 - E) / w
        + b * (1.0 - (1.0 + w * t) * E) / (w * w)
    )


def polynomial(t, p, knots=None, coeffs=None, loop=True):
    """
    Piecewise polynomial from minimum_snap(). coeffs[i] holds the ascending
    power coefficients of segment i."""
    total = knots[-1]
    tau = jnp.mod(t, total) if loop else jnp.clip(t, 0.0, total)
    i = jnp.clip(jnp.searchsorted(knots, tau, side="right") - 1, 0, coeffs.shape[0] - 1)
    s = tau - knots[i]
    # cumprod, not s**arange: the latter differentiates to 0 * s**-1 = NaN at s=0
    n = coeffs.shape[1]
    powers = jnp.concatenate([jnp.ones(1), jnp.cumprod(jnp.full(n - 1, s))])
    return powers @ coeffs[i]


# Heading (yaw) references.
def yaw_fixed(t, p, pos_fn, psi=0.0):
    return jnp.array([jnp.cos(psi), jnp.sin(psi), 0.0])


def yaw_spin(t, p, pos_fn, psi=0.0, rate=0.6):
    a = psi + rate * t
    return jnp.array([jnp.cos(a), jnp.sin(a), 0.0])


def _blend_dir(v, r0, psi0):
    """
    Heading vector that never vanishes.  Aiming the nose at something you fly
    *through* would need unbounded yaw rate, so within a radius r0 of the
    singularity the reference slides smoothly onto the fixed heading psi0
    instead of whipping around.
    """
    d2 = jnp.sum(v * v)
    w = d2 / (d2 + r0 * r0)
    fb = jnp.array([jnp.cos(psi0), jnp.sin(psi0), 0.0])
    return w * v + (1.0 - w) * r0 * fb


def yaw_follow(t, p, pos_fn, r0=0.25, psi0=0.0):
    """Nose along the flight path (guarded at the stall points."""
    v = jax.jacfwd(pos_fn)(t, p)
    return -_blend_dir(jnp.array([v[0], v[1], 0.0]), r0, psi0)


def yaw_center(t, p, pos_fn, target=(0.0, 0.0), r0=0.8, psi0=0.0):
    """
    Nose (and onboard camera) locked on a ground point.  Only sane for paths
    that keep their distance: a curve through the target -- lissajous, rose --
    demands huge yaw torque, so those default to a fixed heading instead.
    """
    r = pos_fn(t, p)
    return -_blend_dir(jnp.array([target[0] - r[0], target[1] - r[1], 0.0]), r0, psi0)


def yaw_from_p(t, p, pos_fn):
    """Live yaw setpoint"""
    a = p["yaw"] + p["yaw_rate"] * t
    return jnp.array([jnp.cos(a), jnp.sin(a), 0.0])


YAW_MODES = {
    "fixed": yaw_fixed,
    "spin": yaw_spin,
    "follow": yaw_follow,
    "center": yaw_center,
    "live": yaw_from_p,
}


# Flat outputs -> full state + feed-forward control
def make_reference_fn(pos_fn, xc_fn):
    """
    Build ref_at_t for a position curve.

    Attitude comes from differential flatness; body rates and angular
    acceleration come from autodiff of the rotation matrix itself, so omega_z
    (which the usual hand-derived formulas drop) is exact.
    """
    d1 = jax.jacfwd(pos_fn, 0)
    d2 = jax.jacfwd(d1, 0)

    def R_fn(t, p):
        zb = _unit(d2(t, p) + g * _E3)  # thrust axis
        yb = _unit(jnp.cross(zb, xc_fn(t, p, pos_fn)))
        return jnp.stack([jnp.cross(yb, zb), yb, zb], axis=1)

    dR = jax.jacfwd(R_fn, 0)

    def omega_fn(t, p):
        R = R_fn(t, p)
        return vee_jnp(R.T @ dR(t, p))  # R^T Rdot = hat(omega_body)

    domega = jax.jacfwd(omega_fn, 0)

    def ref_at_t(t, p):
        R = R_fn(t, p)
        w = omega_fn(t, p)
        thrust = m * jnp.dot(R[:, 2], d2(t, p) + g * _E3)
        tau = _J @ domega(t, p) + jnp.cross(w, _J @ w)
        u_ref = _MIXER_INV @ jnp.concatenate([jnp.array([thrust]), tau])
        x_ref = jnp.concatenate(
            [pos_fn(t, p), rot_to_quat(R), R.T @ d1(t, p), w]  # v in BODY frame
        )
        return x_ref, u_ref

    return ref_at_t


@jit
def align_quaternions(x_refs, q_anchor):
    """
    double cover handling.
    """

    def step(q_prev, x):
        q = quat_align(x[3:7], q_prev)
        return q, x.at[3:7].set(q)

    _, out = jax.lax.scan(step, q_anchor, x_refs)
    return out


# Minimum-snap polynomial through waypoints
def _poly_row(order, deriv, s):
    """Row r with r @ c = d^deriv/dt^deriv (sum_k c_k t^k) evaluated at t = s."""
    row = np.zeros(order + 1)
    for k in range(deriv, order + 1):
        coef = np.prod([k - j for j in range(deriv)]) if deriv else 1.0
        row[k] = coef * s ** (k - deriv)
    return row


def minimum_snap(waypoints, total_time=12.0, loop=True, order=7, min_deriv=4):
    """
    Piecewise polynomial through waypoints minimizing the integral of the
    squared 4th derivative (snap) (the classic quadrotor trajectory, solved
    here as the KKT system of an equality-constrained QP).
    """
    W = np.asarray(waypoints, dtype=float)
    if loop:
        W = np.vstack([W, W[0]])
    dist = np.linalg.norm(np.diff(W, axis=0), axis=1)
    if np.any(dist <= 0):
        raise ValueError("consecutive waypoints must differ")
    T = total_time * dist / dist.sum()
    knots = np.concatenate([[0.0], np.cumsum(T)])
    ns, nc = len(T), order + 1

    H = np.zeros((ns * nc, ns * nc))
    for i, Ti in enumerate(T):
        for a in range(min_deriv, nc):
            for b in range(min_deriv, nc):
                ca = np.prod([a - j for j in range(min_deriv)])
                cb = np.prod([b - j for j in range(min_deriv)])
                pw = a + b - 2 * min_deriv + 1
                H[i * nc + a, i * nc + b] = ca * cb * Ti**pw / pw
    H += 1e-9 * np.eye(ns * nc)  # keep the KKT solve well conditioned

    rows, rhs = [], []

    def row(seg, deriv, s):
        r = np.zeros(ns * nc)
        r[seg * nc : (seg + 1) * nc] = _poly_row(order, deriv, s)
        return r

    for i in range(ns):  # interpolate the waypoints
        rows.append(row(i, 0, 0.0)), rhs.append(W[i])
        rows.append(row(i, 0, T[i])), rhs.append(W[i + 1])
    joints = range(ns) if loop else range(ns - 1)
    for i in joints:  # C1..C3 continuity across joints (wraps when looping)
        j = (i + 1) % ns
        for d in (1, 2, 3):
            rows.append(row(i, d, T[i]) - row(j, d, 0.0)), rhs.append(np.zeros(3))
    if not loop:  # rest to rest
        for d in (1, 2, 3):
            rows.append(row(0, d, 0.0)), rhs.append(np.zeros(3))
            rows.append(row(ns - 1, d, T[-1])), rhs.append(np.zeros(3))

    A = np.array(rows)
    b = np.array(rhs)
    n, mrows = ns * nc, A.shape[0]
    KKT = np.block([[2 * H, A.T], [A, np.zeros((mrows, mrows))]])
    sol = np.linalg.solve(KKT, np.vstack([np.zeros((n, 3)), b]))
    return knots, sol[:n].reshape(ns, nc, 3)


RACE_CIRCUIT = [
    (2.6, 0.0, 1.1),
    (1.4, 2.3, 2.7),
    (-1.4, 2.3, 1.1),
    (-2.6, 0.0, 2.7),
    (-1.4, -2.3, 1.1),
    (1.4, -2.3, 2.7),
]


# Registry -- add a trajectory here and it shows up everywhere
def _twopi_over_f(prm):
    return 2.0 * np.pi / prm["f"]


TRAJECTORIES = {
    "hover": dict(
        fn=hover,
        params=dict(center=(0.0, 0.0, 2.0)),
        yaw="fixed",
        period=lambda prm: 4.0,
        blurb="station keeping",
    ),
    "circle": dict(
        fn=circle,
        params=dict(R=2.0, f=1.0, zd=2.0),
        yaw="center",
        period=_twopi_over_f,
        blurb="constant-speed circle",
    ),
    "figure8": dict(
        fn=figure8,
        params=dict(Rx=3.0, Ry=2.0, f=0.6, zd=2.0),
        yaw="follow",
        period=_twopi_over_f,
        blurb="lemniscate, crosses its own path",
    ),
    "lissajous": dict(
        fn=lissajous,
        params=dict(
            A=(2.5, 2.0, 0.6), n=(2.0, 3.0, 2.0), phase=(0.0, 0.0, 0.0), f=0.5, zd=2.2
        ),
        yaw="fixed",
        period=_twopi_over_f,
        blurb="3D Lissajous weave",
    ),
    "helix": dict(
        fn=helix,
        params=dict(R=2.0, f=1.1, zd=2.0, dz=1.2, fz=0.35),
        yaw="follow",
        period=lambda prm: 2.0 * np.pi / prm["fz"],
        blurb="corkscrew climb and descent",
    ),
    "rose": dict(
        fn=rose,
        params=dict(A=2.4, k=5.0, f=0.3, zd=2.0),
        yaw="fixed",
        period=_twopi_over_f,
        blurb="5-petal polar rose",
    ),
    "trefoil": dict(
        fn=torus_knot,
        params=dict(R=2.0, r=0.7, pw=2.0, qw=3.0, f=0.5, zd=2.2),
        yaw="follow",
        period=_twopi_over_f,
        blurb="(2,3) torus knot -- true 3D loop",
    ),
    "spirograph": dict(
        fn=epitrochoid,
        params=dict(R=1.8, r=0.45, d=0.9, f=0.4, zd=2.0),
        yaw="center",
        period=_twopi_over_f,
        blurb="4-lobe epitrochoid",
    ),
    "heart": dict(
        fn=heart,
        params=dict(A=0.11, f=0.6, zd=2.5, y=0.0),
        yaw="fixed",
        period=_twopi_over_f,
        blurb="heart drawn upright in the x-z plane",
    ),
    "square": dict(
        fn=squircle,
        params=dict(A=2.2, power=4.0, f=0.6, zd=2.0),
        yaw="follow",
        period=_twopi_over_f,
        blurb="rounded square (smooth Lame curve)",
    ),
    "racetrack": dict(
        fn=polynomial,
        params=dict(waypoints=RACE_CIRCUIT, total_time=10.0, loop=True),
        yaw="follow",
        period=lambda prm: prm["total_time"],
        blurb="minimum-snap loop through waypoints",
    ),
    "setpoint": dict(
        fn=setpoint,
        params=dict(),
        yaw="live",
        period=lambda prm: None,
        blurb="live keyboard setpoint",
    ),
}


def _build_pos_fn(name, prm):
    """Bind static shape parameters into a position function."""
    from functools import partial

    fn = TRAJECTORIES[name]["fn"]
    if fn is polynomial:
        knots, coeffs = minimum_snap(
            prm["waypoints"], total_time=prm["total_time"], loop=prm["loop"]
        )
        return partial(
            polynomial,
            knots=jnp.asarray(knots),
            coeffs=jnp.asarray(coeffs),
            loop=prm["loop"],
        )
    return partial(fn, **prm)


def list_trajectories():
    return [f"{k:<11s} {v['blurb']}" for k, v in TRAJECTORIES.items()]


class TrajectoryGenerator:
    """
    Time-warped reference generator.
    """

    def __init__(
        self, traj_type="circle", speed=1.0, yaw_mode=None, yaw_params=None, **params
    ):
        if traj_type not in TRAJECTORIES:
            raise ValueError(
                f"unknown trajectory '{traj_type}'. available:\n  "
                + "\n  ".join(list_trajectories())
            )
        from functools import partial

        spec = TRAJECTORIES[traj_type]
        self.name = traj_type
        self.params = {**spec["params"], **params}
        self.shape_fn = _build_pos_fn(traj_type, self.params)
        self.period = spec["period"](self.params)
        self.warped = traj_type != "setpoint"

        def warped(t, p):
            if not self.warped:
                return self.shape_fn(t, p)
            return self.shape_fn(p["phase"] + p["speed"] * t, p)

        self.pos_fn = warped
        self.yaw_mode = yaw_mode or spec["yaw"]
        if self.yaw_mode not in YAW_MODES:
            raise ValueError(f"unknown yaw mode '{self.yaw_mode}'")
        self.xc_fn = partial(YAW_MODES[self.yaw_mode], **(yaw_params or {}))

        self.ref_at_t = make_reference_fn(self.pos_fn, self.xc_fn)
        self._refs = jit(jax.vmap(self.ref_at_t, in_axes=(0, None)))
        self._tvec = {}

        self.speed = float(speed)
        self.phase = 0.0
        self._t_prev = None
        self.live = {}  # extra run-time params (setpoint mode)

    # --- clock ---
    def reset(self, phase=0.0):
        self.phase = phase
        self._t_prev = None

    def set_speed(self, speed, lo=0.05, hi=4.0):
        self.speed = float(np.clip(speed, lo, hi))
        return self.speed

    def _advance(self, t_sim):
        if self.warped and self._t_prev is not None and t_sim > self._t_prev:
            self.phase += self.speed * (t_sim - self._t_prev)
        self._t_prev = t_sim

    def _p(self):
        return {
            "phase": jnp.asarray(self.phase),
            "speed": jnp.asarray(self.speed),
            **self.live,
        }

    # --- references ---
    def get_refs_batched(self, t_sim, h, Nh, q_anchor=None):
        """(Nh+1, 13) states and (Nh+1, 4) feed-forward thrusts from t_sim on."""
        self._advance(t_sim)
        key = (h, Nh)
        if key not in self._tvec:
            self._tvec[key] = jnp.arange(Nh + 1) * h
        x_refs, u_refs = self._refs(self._tvec[key], self._p())
        # Always chain-align
        anchor = x_refs[0, 3:7] if q_anchor is None else jnp.asarray(q_anchor)
        return align_quaternions(x_refs, anchor), u_refs

    def get_ref(self, t_offset=0.0):
        return self.ref_at_t(jnp.asarray(t_offset), self._p())

    def path_points(self, n=360, laps=1.0):
        """Positions along the reference path, for drawing it in the viewer."""
        if self.period is None:
            return np.zeros((0, 3))
        span = laps * self.period / max(self.speed, 1e-6)
        p = {
            "phase": jnp.asarray(self.phase),
            "speed": jnp.asarray(self.speed),
            **self.live,
        }
        ts = jnp.linspace(0.0, span, n)
        return np.asarray(jax.vmap(self.pos_fn, in_axes=(0, None))(ts, p))

    # --- feasibility ---
    def feasibility(self, n=400, laps=1.0):
        """Peak demands of one lap, and whether the rotors can deliver them."""
        from quad_model import U_MAX, U_MIN

        span = (laps * (self.period or 4.0)) / max(self.speed, 1e-6)
        ts = jnp.linspace(0.0, span, n)
        x, u = jax.vmap(self.ref_at_t, in_axes=(0, None))(ts, self._p())
        x, u = np.asarray(x), np.asarray(u)
        R_z = 1.0 - 2.0 * (x[:, 4] ** 2 + x[:, 5] ** 2)  # cos(tilt) = R[2,2]
        z_min = float(x[:, 2].min())
        return dict(
            z_min=z_min,
            speed=float(np.linalg.norm(x[:, 7:10], axis=1).max()),
            tilt_deg=float(np.degrees(np.arccos(np.clip(R_z, -1, 1))).max()),
            rate=float(np.linalg.norm(x[:, 10:13], axis=1).max()),
            u_min=float(u.min()),
            u_max=float(u.max()),
            ok=bool(
                u.min() >= U_MIN.min() - 1e-6
                and u.max() <= U_MAX.max() + 1e-6
                and z_min > 0.15
            ),  # a reference under the floor is untrackable
        )

    def report(self):
        fs = self.feasibility()
        per = "-" if self.period is None else f"{self.period / self.speed:5.2f} s"
        return (
            f"trajectory '{self.name}' yaw={self.yaw_mode} speed={self.speed:.2f} "
            f"lap={per}\n"
            f"  peak: |v| {fs['speed']:5.2f} m/s   tilt {fs['tilt_deg']:5.1f} deg   "
            f"|w| {fs['rate']:5.2f} rad/s   rotor [{fs['u_min']:5.2f}, "
            f"{fs['u_max']:5.2f}] N   z_min {fs['z_min']:5.2f} m  -> "
            f"{'feasible' if fs['ok'] else 'NOT FLYABLE'}"
        )


if __name__ == "__main__":
    for name in TRAJECTORIES:
        if name == "setpoint":
            continue
        print(TrajectoryGenerator(name).report())
