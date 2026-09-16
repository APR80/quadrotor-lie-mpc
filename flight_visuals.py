from dataclasses import dataclass
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw


@dataclass
class DisplayOptions:
    trail: bool = True
    hud: bool = True
    keys: bool = True
    paths: bool = True
    disturb_arrow: bool = True
    clean: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, name) for name in cls.__dataclass_fields__})

    def visible(self, name):
        return not self.clean and getattr(self, name)

    def handle_key(self, key):
        name = {
            glfw.KEY_F1: "hud",
            glfw.KEY_F2: "keys",
            glfw.KEY_F3: "trail",
            glfw.KEY_F4: "clean",
            glfw.KEY_F5: "disturb_arrow",
            glfw.KEY_F6: "paths",
        }.get(key)
        if name is not None:
            setattr(self, name, not getattr(self, name))
            return True
        return False


def add_display_arguments(ap):
    group = ap.add_argument_group("display")
    for flag, dest, description in (
        ("trail", "trail", "yellow flight trail (F3)"),
        ("hud", "hud", "top-left guide and flight statistics (F1)"),
        ("keys", "keys", "keyboard overlay (F2)"),
        ("paths", "paths", "reference path and command line (F6)"),
        ("disturb-arrow", "disturb_arrow", "disturbance vector (F5)"),
    ):
        group.add_argument(
            f"--{flag}",
            dest=dest,
            action="store_true",
            default=True,
            help=f"show {description} (default)",
        )
        group.add_argument(
            f"--no-{flag}", dest=dest, action="store_false", help=f"hide {description}"
        )
    group.add_argument(
        "--clean",
        action="store_true",
        help="hide all overlays and guide lines; F4 toggles while flying",
    )


def set_force_arrow(
    arrows,
    name,
    tip,
    vec,
    metres_per_newton=0.08,
    rgba=(1.0, 0.3, 0.18, 1.0),
    width=0.003,
):
    """vector pointing out from the vehicle in the force direction."""
    vec = None if vec is None else np.asarray(vec, dtype=np.float64)
    if vec is None or np.linalg.norm(vec) < 1e-8:
        arrows.pop(name, None)
        return
    tip = np.asarray(tip, dtype=np.float64).copy()
    arrows[name] = (tip, tip + vec * metres_per_newton, rgba, width)


def draw_force_arrow(scene, a, b, rgba, width):
    delta = b - a
    length = np.linalg.norm(delta)
    if length < 1e-8 or scene.ngeom + 3 > scene.maxgeom:
        return
    direction = delta / length
    eye = (scene.camera[0].pos + scene.camera[1].pos) * 0.5
    side = np.cross(direction, eye - b)
    if np.linalg.norm(side) < 1e-8:
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        side = np.cross(direction, axis)
    side /= np.linalg.norm(side)
    head = min(0.16, length * 0.28)
    base = b - direction * head
    for start, end in (
        (a, b),
        (base + side * head * 0.5, b),
        (base - side * head * 0.5, b),
    ):
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            mj.mjtGeom.mjGEOM_CYLINDER,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.asarray(rgba, dtype=np.float32),
        )
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CYLINDER, width, start, end)
        geom.emission = 1.0
        geom.specular = 0.0
        geom.category = mj.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def draw_keyboard(viewport, context, keys):
    """draw directly into the framebuffer"""
    size = max(32, min(48, viewport.width // 27, viewport.height // 12))
    gap = max(2, size // 12)
    margin = max(8, size // 3)
    left = viewport.left + viewport.width - 3 * (size + gap) - margin
    top = viewport.bottom + viewport.height - margin
    layout = (
        (0, 1, 1, "^", (glfw.KEY_UP,)),
        (1, 0, 1, "<", (glfw.KEY_LEFT,)),
        (1, 1, 1, "v", (glfw.KEY_DOWN,)),
        (1, 2, 1, ">", (glfw.KEY_RIGHT,)),
        (2, 1, 1, "W", (glfw.KEY_W,)),
        (3, 0, 1, "A", (glfw.KEY_A,)),
        (3, 1, 1, "S", (glfw.KEY_S,)),
        (3, 2, 1, "D", (glfw.KEY_D,)),
        (4, 0, 2, "SHIFT", (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)),
        (6, 0, 1, "U", (glfw.KEY_U,)),
        (6, 1, 1, "I", (glfw.KEY_I,)),
        (6, 2, 1, "O", (glfw.KEY_O,)),
        (7, 0, 1, "J", (glfw.KEY_J,)),
        (7, 1, 1, "K", (glfw.KEY_K,)),
        (7, 2, 1, "L", (glfw.KEY_L,)),
    )
    for row, col, span, label, codes in layout:
        pressed = any(keys.get(code, False) for code in codes)
        bg = (
            ((1.0, 0.28, 0.18) if row >= 6 else (0.0, 0.78, 1.0))
            if pressed
            else (0.16, 0.18, 0.21)
        )
        rect = mj.MjrRect(
            left + col * (size + gap),
            top - (row + 1) * (size + gap),
            span * size + (span - 1) * gap,
            size,
        )
        mj.mjr_label(
            rect, mj.mjtFont.mjFONT_NORMAL, label, *bg, 1.0, 1.0, 1.0, 1.0, context
        )
