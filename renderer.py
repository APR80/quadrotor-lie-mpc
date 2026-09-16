from collections import defaultdict, deque
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
from flight_visuals import (
    DisplayOptions,
    draw_force_arrow,
    draw_keyboard,
    set_force_arrow,
)


class MujocoRenderer:
    def __init__(
        self,
        model,
        data,
        height=900,
        width=1200,
        title="MuJoCo Simulation",
        follow=False,
        follow_body=None,
        display=None,
        frame_callback=None,
    ):
        self.model = model
        self.data = data
        self.display = display if display is not None else DisplayOptions(keys=False)
        self.frame_callback = frame_callback

        self._button_left = self._button_middle = self._button_right = False
        self._lastx = self._lasty = 0

        # keys: held state for any key, plus a drainable queue of fresh presses
        self.keys = defaultdict(bool)
        self._pressed = deque()

        glfw.init()
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.scene = mj.MjvScene(self.model, maxgeom=20000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        mj.mjv_defaultCamera(self.cam)
        self.cam.lookat = np.array([0.0, 0.0, 1.0])
        self.cam.distance = 6.0
        self.cam.azimuth = 90.0
        self.cam.elevation = -20.0

        self.follow = follow
        self.follow_body = (
            model.body(follow_body).id if isinstance(follow_body, str) else follow_body
        )
        self.follow_gain = 0.12

        self.hud = ""
        self.hud_right = ""
        self.paths = {}
        self.arrows = {}
        self._trail = deque(maxlen=1500)
        self.trail_rgba = (1.0, 0.85, 0.1, 1.0)

        glfw.set_key_callback(self.window, self._keyboard_callback)
        glfw.set_window_focus_callback(self.window, self._focus_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

    # --- input ---
    def _keyboard_callback(self, window, key, scancode, act, mods):
        if act == glfw.PRESS:
            self.keys[key] = True
            self._pressed.append(key)
        elif act == glfw.RELEASE:
            self.keys[key] = False

    def take_pressed(self):
        """Keys pressed since the last call (edge triggered), then clear."""
        out = list(self._pressed)
        self._pressed.clear()
        return out

    def _focus_callback(self, window, focused):
        if not focused:
            self.keys.clear()  # released keys outside the window must not keep pushing

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self._button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        self._lastx, self._lasty = glfw.get_cursor_pos(window)

    def _mouse_move_callback(self, window, xpos, ypos):
        dx, dy = xpos - self._lastx, ypos - self._lasty
        self._lastx, self._lasty = xpos, ypos
        if not (self._button_left or self._button_middle or self._button_right):
            return
        _, height = glfw.get_window_size(window)
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left:
            action = (
                mj.mjtMouse.mjMOUSE_ROTATE_H if shift else mj.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene, self.cam
        )

    def _scroll_callback(self, window, xoffset, yoffset):
        mj.mjv_moveCamera(
            self.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.cam,
        )

    # --- overlays ---
    def set_hud(self, text, right=""):
        self.hud, self.hud_right = text, right

    def set_path(self, name, points, rgba=(0.2, 0.8, 1.0, 1.0), width=0.012):
        """Register a polyline to draw every frame (None/empty removes it)."""
        if points is None or len(points) < 2:
            self.paths.pop(name, None)
            return
        self.paths[name] = (np.asarray(points, dtype=np.float64), rgba, width)

    def push_trail(self, point, every=3):
        """Record where the vehicle actually went, subsampled."""
        self._n_trail = getattr(self, "_n_trail", 0) + 1
        if self._n_trail % every == 0:
            self._trail.append(np.array(point, dtype=np.float64))

    def set_arrow(self, name, tip, vec, **kwargs):
        set_force_arrow(self.arrows, name, tip, vec, **kwargs)

    def clear_trail(self):
        self._trail.clear()

    def _draw_line(self, pts, rgba, width):
        rgba = np.array(rgba, dtype=np.float32)
        size, pos = np.zeros(3), np.zeros(3)
        mat = np.eye(3).ravel()
        for a, b in zip(pts[:-1], pts[1:]):
            if self.scene.ngeom >= self.scene.maxgeom:
                return
            geom = self.scene.geoms[self.scene.ngeom]
            mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, size, pos, mat, rgba)
            mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, width, a, b)
            geom.category = mj.mjtCatBit.mjCAT_DECOR
            self.scene.ngeom += 1

    # --- frame ---
    def render(self):
        w, hgt = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, w, hgt)

        if self.follow and self.follow_body is not None:
            target = self.data.xipos[self.follow_body]
            self.cam.lookat[:] += self.follow_gain * (target - self.cam.lookat)

        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        if self.display.visible("paths"):
            for pts, rgba, width in self.paths.values():
                self._draw_line(pts, rgba, width)
        if self.display.visible("trail") and len(self._trail) > 1:
            self._draw_line(np.array(self._trail), self.trail_rgba, 0.008)
        if self.display.visible("disturb_arrow"):
            for arrow in self.arrows.values():
                draw_force_arrow(self.scene, *arrow)

        mj.mjr_render(viewport, self.scene, self.context)
        if self.display.visible("hud") and (self.hud or self.hud_right):
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL.value,
                mj.mjtGridPos.mjGRID_TOPLEFT.value,
                viewport,
                self.hud,
                self.hud_right,
                self.context,
            )
        if self.display.visible("keys"):
            draw_keyboard(viewport, self.context, self.keys)
        if self.frame_callback is not None:
            self.frame_callback(self, viewport)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def is_window_closed(self):
        return glfw.window_should_close(self.window)

    def close(self):
        glfw.terminate()
