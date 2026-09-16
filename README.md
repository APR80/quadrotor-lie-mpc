# quadrotor-lie-mpc
**Tracking MPC for a 3D quadrotor, with singularity-free attitude optimization on the Lie group of rotations.**

This project started as my attempt at a fast tracking-MPC solver built with JAX and OSQP. The interesting part turned out to be the attitude handling: instead of parameterizing orientation with Euler angles (which suffer from gimbal lock and numerical problems), I used the idea from *Planning With Attitude* (Jackson et al.), where the optimizer works in the **12-dimensional error state** on the tangent space of the unit quaternion, lifted with the **attitude Jacobian** `G(q)`. This approach is singularity-free, computationally efficient, and costs almost nothing: `G(q)^T G(q) = I/4` for a unit quaternion, so the pseudo-inverse is `4 G(q)^T` in closed form. No SVD, no `pinv` evaluation every tick.

The plant is the Skydio X2 from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2) (1.325 kg, thrust/weight 4.0). Mass, inertia, and the thrust/torque mixer are all read out of the model, not hardcoded.

# How fast is it

Here is a measured run on an **Intel Core i5-12450H**, Linux, Python 3.11.15, with **JAX on CPU** and its default 32-bit precision:

```bash
python tracking_mpc_optimized.py --headless --traj figure8 --Nh 20 --speed 1 --duration 11
```

The control interval is **10 ms (100 Hz)**. A horizon of 20 steps looks ahead **0.20 s** and gives a QP with **320 decision variables** and **320 constraint rows**, including rotor bounds. The run covers slightly more than one full figure-eight lap.

| Measurement | Result |
| --- | ---: |
| Mean controller update | **2.84 ms** |
| 95th-percentile controller update | **4.06 ms** |
| Maximum controller update | **9.04 ms** |
| Mean OSQP iterations | **44** |
| QP solves / failures | **1,100 / 0** |
| RMS position error | **1.0 cm** |
| Maximum position error | **1.7 cm** |
| RMS attitude error, small-angle approximation | **0.7°** |
| Maximum attitude error, small-angle approximation | **1.2°** |

These are measurements from one local run. The timer includes reference generation, JAX linearization, transfer to NumPy, QP updates, periodic terminal-cost computation, and the OSQP solve. It excludes MuJoCo stepping and rendering.

Every timed controller update after the first stayed within the 10 ms control period in this run, with a measured maximum of 9.04 ms. The simulation advances in fixed steps regardless of wall-clock timing, so these are not a hard real-time deadline guarantee.


The reason behind these numbers are batched JAX autodiff and JIT compilation, and the real win coming from the fact that sparsity pattern never changes between ticks, only the numerical values, so OSQP is set up once and updated in place through precomputed CSC index maps: no allocation, no re-factorization of the KKT structure, warm-started from the last solution.

# Getting started

Use a Python environment with these dependencies with the versions used for the measurements above; I used Python 3.11.

```bash
python -m pip install \
  "jax[cpu]==0.10.2" "jaxlib==0.10.1" \
  "numpy==2.4.6" "scipy==1.17.1" \
  "osqp==1.1.3" "control==0.10.2" \
  "mujoco==3.10.0" "glfw==2.10.0"
```

Run commands from the repository root. Keep `google-deepmind mujoco_menagerie main skydio_x2` directory alongside the Python files.

```bash
# Interactive viewer
python tracking_mpc_optimized.py

# for a headless run
python tracking_mpc_optimized.py --headless --traj hover --duration 10

# for a specific trajectory
python tracking_mpc_optimized.py --traj heart --speed 1

# fly it yourself
python keyboard_flight.py

# command-line options
python tracking_mpc_optimized.py --help
```

Available trajectories are `hover`, `circle`, `figure8`, `lissajous`, `helix`, `rose`, `trefoil`, `spirograph`, `heart`, `square`, and `racetrack`. Position curves and heading references feed a differential-flatness generator that derives attitude, body velocity, body rates, and feed-forward rotor thrusts. The startup report samples the reference to estimate its thrust, tilt, rate, and altitude demands.


# Attitude handling

The full state has 13 components:

```text
x = [position_world (3), quaternion_wxyz (4), velocity_body (3), angular_velocity_body (3)]
```

Position and velocity describe the center of mass. The quaternion maps body coordinates to world coordinates. The optimizer uses a 12-component error state:

```text
dx = [position_error (3), attitude_error (3), velocity_error (3), angular_velocity_error (3)]
```

`E(q)` lifts this error state into the full-state tangent, using `G(q)` for the attitude block. JAX differentiates the RK4 dynamics, and the lifted/projected Jacobians give the linear error dynamics along the reference. The QP also includes the reference's discrete dynamics defect, rotor thrust bounds, and an LQR terminal cost. Only the first control is applied before solving again.

Quaternions are normalized during integration and reference signs are aligned across the horizon to handle the double cover: `q` and `-q` represent the same rotation.


# An Important Note
Here, **singularity-free** refers to the attitude representation and full-rank tangent lift. The MPC still uses a local, first-order approximation but this controller still performs very well.

I ran an initially inverted attitude with quaternion **`[0, 1, 0, 0]`**(180° rotation about the body x-axis relative to the upright reference) the controller can still recover to upright hover..

Why does the linearization not automatically fail for such large error? For every unit quaternion, `G(q)^T G(q) = I/4`, including at 180°. The attitude Jacobian therefore retains full column rank. The implemented attitude error is

```text
e_att = 4 G(q_ref)^T (q - q_ref)
```

For this initial state, `e_att = [2, 0, 0]`: a finite, nonzero signal that lets the controller request a corrective torque. More generally, after quaternion sign alignment, its magnitude is `2 sin(theta/2)`, where `theta` is the actual angular error. This is approximately `theta` near the reference, but equals `2` at 180°, rather than `pi`. The local prediction is inaccurate for such a large error, yet the magic of feedback and constrained QP solves can still bring it into the neighborhood where that prediction is accurate.


# Code 

| File | Purpose |
| --- | --- |
| [`quad_model.py`](quad_model.py) | Plant parameters, quaternion helpers, RK4 dynamics, and MuJoCo state conversion |
| [`tracking_mpc_optimized.py`](tracking_mpc_optimized.py) | Sparse error-state QP, controller, and tracking demo |
| [`trajectory_generator.py`](trajectory_generator.py) | Differential-flatness references and minimum-snap trajectories |
| [`keyboard_flight.py`](keyboard_flight.py) | Keyboard-driven velocity-setpoint demo |
| [`disturbances.py`](disturbances.py) | Scripted forces, torques, wind, gusts, and keyboard pushes |
| [`renderer.py`](renderer.py) / [`flight_visuals.py`](flight_visuals.py) | Viewer, trails, overlays, and force arrows |
