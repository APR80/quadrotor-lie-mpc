"""
The quadrotor plant.
Everything that both the tracking MPC and the keyboard demo need lives here:
the MuJoCo handle, the physical parameters read out of the model.
"""

import os
# This is for my laptop. you may comment it out.
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")  # NVIDIA Optimus/PRIME
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # tiny matrices: threads hurt
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
from jax import jacfwd, jit

_HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(
    _HERE, "google-deepmind mujoco_menagerie main skydio_x2", "scene.xml"
)

BODY_NAME = "x2"

model = mj.MjModel.from_xml_path(XML_PATH)
body_id = model.body(BODY_NAME).id


def new_data():
    """Fresh MjData for model (each demo owns its own simulation state)."""
    return mj.MjData(model)


# --- Physical parameters read from the model ---
h = float(model.opt.timestep)  # 0.01 s
g = float(-model.opt.gravity[2])  # 9.81
m = float(mj.mj_getTotalmass(model))  # 1.325 kg
com_body = model.body_ipos[body_id].copy()  # COM in the body frame

# body_inertia is diagonal in the PRINCIPAL frame given by body_iquat, which for
# this model is a permutation of the body axes.
_Rp = np.zeros(9)
mj.mju_quat2Mat(_Rp, model.body_iquat[body_id])
_Rp = _Rp.reshape(3, 3)
J = _Rp @ np.diag(model.body_inertia[body_id]) @ _Rp.T
J = 0.5 * (J + J.T)  # kill round-off asymmetry
Jinv = np.linalg.inv(J)


def _build_mixer():
    """[T; tau_body] = MIXER @ u, derived from the actuator sites and gears."""
    M = np.zeros((4, 4))
    for i in range(model.nu):
        site = model.actuator_trnid[i, 0]
        arm = model.site_pos[site] - com_body  # lever arm about the COM
        gear = model.actuator_gear[i]
        kf, kz = gear[2], gear[5]  # body-z force, body-z reaction torque
        M[0, i] = kf
        M[1:, i] = np.cross(arm, np.array([0.0, 0.0, kf])) + np.array([0.0, 0.0, kz])
    return M


MIXER = _build_mixer()
MIXER_INV = np.linalg.inv(MIXER)

kt = float(MIXER[0, 0])  # thrust per unit ctrl  [N]
km = float(abs(MIXER[3, 0]))  # yaw torque per unit ctr
U_MIN = model.actuator_ctrlrange[:, 0].copy()  # 0 N
U_MAX = model.actuator_ctrlrange[:, 1].copy()  # 13 N
U_HOVER = MIXER_INV @ np.array([m * g, 0.0, 0.0, 0.0])

NX_FULL, NX, NU = 13, 12, 4  # full state, error state, inputs

MIXER_jnp = jnp.asarray(MIXER)
J_jnp = jnp.asarray(J)
Jinv_jnp = jnp.asarray(Jinv)
_com_jnp = jnp.asarray(com_body)


# --- Quaternion helpers ---
def hat(v):
    """Skew-symmetric matrix, numpy."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def q_to_R(q):
    """Unit quaternion (w,x,y,z) -> rotation matrix (body -> world), numpy."""
    s, v = q[0], q[1:4]
    return (s**2 - np.dot(v, v)) * np.eye(3) + 2 * np.outer(v, v) + 2 * s * hat(v)


@jit
def hat_jnp(v):
    return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@jit
def vee_jnp(M):
    """Inverse of hat(): pulls the rotation vector out of a skew matrix."""
    return jnp.array([M[2, 1], M[0, 2], M[1, 0]])


@jit
def L_jnp(q):
    """Left quaternion product matrix: L(q) @ p == q * p."""
    s, v = q[0], q[1:4]
    L = jnp.zeros((4, 4))
    L = L.at[0, 0].set(s)
    L = L.at[0, 1:].set(-v)
    L = L.at[1:, 0].set(v)
    L = L.at[1:, 1:].set(s * jnp.eye(3) + hat_jnp(v))
    return L


@jit
def q_to_R_jnp(q):
    s, v = q[0], q[1:4]
    return (
        (s**2 - jnp.dot(v, v)) * jnp.eye(3) + 2 * jnp.outer(v, v) + 2 * s * hat_jnp(v)
    )


@jit
def G(q):
    """d q / d phi: maps a body-frame rotation vector to a quaternion tangent."""
    qw, qx, qy, qz = q
    return 0.5 * jnp.array(
        [[-qx, -qy, -qz], [qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw]]
    )


@jit
def build_E(q):
    """13x12 lift from the error state to the full state tangent."""
    E = jnp.zeros((NX_FULL, NX))
    E = E.at[0:3, 0:3].set(jnp.eye(3))
    E = E.at[3:7, 3:6].set(G(q))
    E = E.at[7:10, 6:9].set(jnp.eye(3))
    E = E.at[10:13, 9:12].set(jnp.eye(3))
    return E


@jit
def build_E_pinv(q):
    """
    12x13 pseudo-inverse of build_E, in closed form.
    """
    E = jnp.zeros((NX, NX_FULL))
    E = E.at[0:3, 0:3].set(jnp.eye(3))
    E = E.at[3:6, 3:7].set(4.0 * G(q).T)
    E = E.at[6:9, 7:10].set(jnp.eye(3))
    E = E.at[9:12, 10:13].set(jnp.eye(3))
    return E


@jit
def quat_align(q_ref, q_like):
    """Flip q_ref onto the hemisphere of q_like (q and -q are the same rotation)."""
    return jnp.where(jnp.dot(q_ref, q_like) < 0.0, -q_ref, q_ref)


# --- Dynamics ---
@jit
def quad_dynamics(x, u):
    """Continuous-time rigid-body dynamics, body-frame velocity convention."""
    q, v, w = x[3:7], x[7:10], x[10:13]
    q = q / jnp.linalg.norm(q)
    R = q_to_R_jnp(q)
    T_tau = MIXER_jnp @ u

    r_dot = R @ v
    q_dot = 0.5 * L_jnp(q) @ jnp.concatenate((jnp.zeros(1), w))
    v_dot = (
        R.T @ jnp.array([0.0, 0.0, -g])  # gravity, body frame
        + jnp.array([0.0, 0.0, T_tau[0] / m])  # collective thrust
        - hat_jnp(w) @ v  # Coriolis (rotating frame)
    )
    w_dot = Jinv_jnp @ (T_tau[1:] - hat_jnp(w) @ (J_jnp @ w))
    return jnp.concatenate((r_dot, q_dot, v_dot, w_dot))


@jit
def quad_dynamics_rk4(x, u):
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5 * h * f1, u)
    f3 = quad_dynamics(x + 0.5 * h * f2, u)
    f4 = quad_dynamics(x + h * f3, u)
    x_next = x + (h / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
    return x_next.at[3:7].set(x_next[3:7] / jnp.linalg.norm(x_next[3:7]))


jac_A = jit(jacfwd(quad_dynamics_rk4, 0))
jac_B = jit(jacfwd(quad_dynamics_rk4, 1))
A_stack_vmap = jit(jax.vmap(jac_A, in_axes=(0, 0)))
B_stack_vmap = jit(jax.vmap(jac_B, in_axes=(0, 0)))
E_stack_vmap = jit(jax.vmap(build_E, in_axes=0))
E_pinv_stack_vmap = jit(jax.vmap(build_E_pinv, in_axes=0))


def mj_to_x(data):
    """MuJoCo qpos/qvel -> 13-state about the COM."""
    q = data.qpos[3:7] / np.linalg.norm(data.qpos[3:7])
    R = q_to_R(q)
    w = data.qvel[3:6]  # already body frame
    r = data.qpos[0:3] + R @ com_body
    v_body = R.T @ data.qvel[0:3] + np.cross(w, com_body)
    return np.concatenate([r, q, v_body, w])


def x_to_mj(data, x):
    """Write a 13-state about the COM into MuJoCo's qpos/qvel"""
    q = x[3:7] / np.linalg.norm(x[3:7])
    R = q_to_R(q)
    w = x[10:13]
    data.qpos[0:3] = x[0:3] - R @ com_body
    data.qpos[3:7] = q
    data.qvel[0:3] = R @ (x[7:10] - np.cross(w, com_body))
    data.qvel[3:6] = w


def describe():
    lines = [
        f"model      : {os.path.relpath(XML_PATH, _HERE)}",
        f"mass       : {m:.4f} kg   g = {g:.2f}   dt = {h * 1e3:.1f} ms",
        f"inertia    : diag {np.diag(J).round(6)}  off-diag max "
        f"{np.abs(J - np.diag(np.diag(J))).max():.2e}",
        f"mixer kt/km: {kt:.4f} / {km:.4f}",
        f"ctrl range : [{U_MIN[0]:.1f}, {U_MAX[0]:.1f}] N per rotor  "
        f"(hover {U_HOVER[0]:.3f}, thrust/weight {U_MAX.sum() / (m * g):.2f})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
