import sys
import numpy as np
from quad_model import g, m

WEIGHT = m * g


def _parse_vec3(text, ctx):
    """'1,2,3' -> [1,2,3];  '4' -> [4,0,0]."""
    try:
        vals = [float(p) for p in str(text).split(",") if p.strip()]
    except ValueError:
        sys.exit(f"{ctx}: {text!r} is not a number or an x,y,z triple")
    if len(vals) == 1:
        return np.array([vals[0], 0.0, 0.0])
    if len(vals) == 3:
        return np.array(vals)
    sys.exit(f"{ctx}: want x,y,z (or one number, taken as x): got {text!r}")


class Disturbance:
    def __init__(
        self, body, events=(), wind=(0.0, 0.0, 0.0), gust=0.0, gust_tau=0.5, seed=0
    ):
        self.body = body
        self.events = sorted(events, key=lambda e: e["t"])
        self.wind = np.asarray(wind, dtype=float).copy()
        self.gust = float(gust)
        self.gust_tau = max(float(gust_tau), 1e-3)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.hold = np.zeros(3)  # live mode: force held down on the keyboard
        self.f = np.zeros(3)
        self.tau = np.zeros(3)
        self.label = ""
        self.peak = 0.0
        self._turb = np.zeros(3)
        self._t_prev = None

    def reset(self):
        """Restart the disturbance clock and seeded turbulence with the vehicle."""
        self.hold[:] = 0.0
        self.f = np.zeros(3)
        self.tau = np.zeros(3)
        self._turb[:] = 0.0
        self._t_prev = None
        self.rng = np.random.default_rng(self.seed)
        self.label = ""
        self.peak = 0.0

    def set_hold(self, keys, strength):
        from mujoco.glfw import glfw

        self.hold = (
            strength
            * WEIGHT
            * np.array(
                [
                    float(keys.get(positive, False)) - float(keys.get(negative, False))
                    for positive, negative in (
                        (glfw.KEY_I, glfw.KEY_K),
                        (glfw.KEY_J, glfw.KEY_L),
                        (glfw.KEY_U, glfw.KEY_O),
                    )
                ]
            )
        )

    # --- spec parsing ---
    @staticmethod
    def parse_event(spec):
        """One --disturb string -> a normalized event dict."""
        e = {
            "t": None,
            "dur": 0.10,
            "f": np.zeros(3),
            "tau": np.zeros(3),
            "frame": "world",
            "shape": "step",
        }
        for tok in spec.replace(";", " ").split():
            if "=" not in tok:
                sys.exit(
                    f"--disturb: don't understand {tok!r} in {spec!r}. "
                    'Use key=value, e.g. "t=2 fw=0.6,0,0 dur=0.2"'
                )
            key, val = (s.strip() for s in tok.split("=", 1))
            key = key.lower()
            if key in ("t", "at"):
                e["t"] = float(val)
            elif key in ("dur", "d"):
                e["dur"] = float(val)
            elif key == "f":
                e["f"] = e["f"] + _parse_vec3(val, "--disturb f")
            elif key == "fw":  # in multiples of the vehicle's weight
                e["f"] = e["f"] + WEIGHT * _parse_vec3(val, "--disturb fw")
            elif key == "tau":
                e["tau"] = e["tau"] + _parse_vec3(val, "--disturb tau")
            elif key == "taum":  # weight * metre: "N g on a 1 m lever arm"
                e["tau"] = e["tau"] + WEIGHT * _parse_vec3(val, "--disturb taum")
            elif key == "frame":
                if val.lower() not in ("world", "body"):
                    sys.exit("--disturb frame= must be world or body")
                e["frame"] = val.lower()
            elif key == "shape":
                if val.lower() not in ("step", "smooth"):
                    sys.exit("--disturb shape= must be step or smooth")
                e["shape"] = val.lower()
            else:
                sys.exit(
                    f"--disturb: unknown key {key!r}. Known: t dur f fw "
                    "tau taum frame shape"
                )
        if e["t"] is None:
            sys.exit(f'--disturb: {spec!r} needs a start time, e.g. "t=2"')
        if e["dur"] <= 0.0:
            sys.exit("--disturb: dur must be > 0")
        if not (np.any(e["f"]) or np.any(e["tau"])):
            sys.exit(f"--disturb: {spec!r} applies no force and no torque")
        return e

    # --- state ---
    @property
    def armed(self):
        return bool(self.events) or bool(np.any(self.wind)) or self.gust > 0.0

    @property
    def last_end(self):
        return max((e["t"] + e["dur"] for e in self.events), default=0.0)

    def plan(self):
        lines = []
        for e in self.events:
            what = []
            if np.any(e["f"]):
                what.append(
                    f"f {np.round(e['f'], 2)} N "
                    f"({np.linalg.norm(e['f']) / WEIGHT:.2f} x weight)"
                )
            if np.any(e["tau"]):
                what.append(f"tau {np.round(e['tau'], 3)} N m")
            lines.append(
                f"  t {e['t']:6.2f} s  +{e['dur']:.2f} s  "
                f"{e['frame']:<5s} {e['shape']:<6s} " + "  ".join(what)
            )
        if np.any(self.wind):
            lines.append(f"  wind  {np.round(self.wind, 2)} N (constant)")
        if self.gust > 0.0:
            lines.append(
                f"  gust  sigma {self.gust:.2f} N, tau "
                f"{self.gust_tau:.2f} s (OU turbulence)"
            )
        return "disturbances:\n" + "\n".join(lines) if lines else "disturbances: none"

    def step(self, data):
        """
        write the step's force/torque into xfrc_applied.
        """
        t = float(data.time)
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t

        f = self.wind + self.hold
        tau = np.zeros(3)
        tags = []
        if np.any(self.hold):
            tags.append("hand")
        if np.any(self.wind):
            tags.append("wind")

        if self.gust > 0.0 and dt > 0.0:
            # Ornstein-Uhlenbeck: dx = -x dt/tau + sigma sqrt(2 dt/tau) dW.
            a = dt / self.gust_tau
            self._turb += -a * self._turb + self.gust * np.sqrt(
                2.0 * a
            ) * self.rng.standard_normal(3)
            f = f + self._turb
            tags.append("gust")

        for e in self.events:
            if not (e["t"] <= t < e["t"] + e["dur"]):
                continue
            w = 1.0
            if e["shape"] == "smooth":  # raised cosine, 0 -> 1 -> 0
                s = (t - e["t"]) / e["dur"]
                w = 0.5 * (1.0 - np.cos(2.0 * np.pi * s))
            ef, et = w * e["f"], w * e["tau"]
            if e["frame"] == "body":
                R = data.xmat[self.body].reshape(3, 3)
                ef, et = R @ ef, R @ et
            f, tau = f + ef, tau + et
            tags.append("push")

        data.xfrc_applied[self.body, 0:3] = f
        data.xfrc_applied[self.body, 3:6] = tau
        self.f, self.tau = f, tau
        self.label = "+".join(dict.fromkeys(tags))
        self.peak = max(self.peak, float(np.linalg.norm(f)))
        return f, tau

    def hud(self):
        if not (np.any(self.f) or np.any(self.tau)):
            return "push       -"
        n = float(np.linalg.norm(self.f))
        line = f"push {n:6.2f} N ({n / WEIGHT:4.2f} g) {self.label}"
        if np.any(self.tau):
            line += f"\ntorque {np.linalg.norm(self.tau):5.2f} N m"
        return line

    def report(self, worst, t_worst, t_recovered):
        out = [
            f"disturbance: peak |f| {self.peak:5.2f} N "
            f"({self.peak / WEIGHT:.2f} x weight)",
            f"  worst position error {worst * 100:5.1f} cm at t {t_worst:5.2f} s",
        ]
        if self.events and t_recovered is not None:
            line = f"  last outside 5 cm at t {t_recovered:5.2f} s"
            if t_recovered > self.last_end:
                line += (
                    f" -- {t_recovered - self.last_end:.2f} s of recovery "
                    f"after the final push ended"
                )
            out.append(line)
        return "\n".join(out)


def add_disturbance_arguments(ap):
    dg = ap.add_argument_group(
        "disturbances",
        "shove the vehicle mid-take, the way ctrl+drag does in mujoco.viewer: "
        "everything is written into data.xfrc_applied, so the MPC only ever "
        "sees the consequences",
    )
    dg.add_argument(
        "--disturb",
        action="append",
        metavar="SPEC",
        default=None,
        help="one scripted push, repeatable. Keys: t (start, s), "
        "dur (s, default 0.1), f (N, world x,y,z), fw (the "
        "same in multiples of weight), tau (N m), taum "
        "(weight*m), frame=world|body, shape=step|smooth. "
        'e.g. --disturb "t=2 fw=0.6,0,0 dur=0.2"',
    )
    dg.add_argument(
        "--wind",
        default=None,
        metavar="X,Y,Z",
        help="constant world force in N for the whole take",
    )
    dg.add_argument(
        "--gust",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="turbulence std in N (Ornstein-Uhlenbeck)",
    )
    dg.add_argument(
        "--gust-tau", type=float, default=0.5, help="turbulence correlation time, s"
    )
    dg.add_argument(
        "--seed", type=int, default=0, help="turbulence seed -- same seed, same video"
    )
    dg.add_argument(
        "--push",
        type=float,
        default=0.5,
        help="live mode: I/K J/L U/O shove strength, in multiples "
        "of the vehicle's weight",
    )
    dg.add_argument(
        "--disturb-scale",
        type=float,
        default=1.0,
        help="force arrow length in metres per weight of force",
    )


def disturbance_from_args(args, body):
    events = [Disturbance.parse_event(s) for s in (args.disturb or [])]
    wind = _parse_vec3(args.wind, "--wind") if args.wind else (0.0, 0.0, 0.0)
    return Disturbance(
        body, events, wind=wind, gust=args.gust, gust_tau=args.gust_tau, seed=args.seed
    )
