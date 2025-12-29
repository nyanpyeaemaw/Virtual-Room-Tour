# engine/character_controller.py
# Drop-in first-person walker with AABB collisions for PyGame + PyOpenGL.
# - Walk on XZ only (no flying) when lock_y=True
# - Y (eye height) stays constant unless gravity>0 and lock_y=False

from dataclasses import dataclass
from typing import List, Tuple
import math

Vec3 = Tuple[float, float, float]
AABB = Tuple[Vec3, Vec3]

def aabb_overlap(a: AABB, b: AABB) -> bool:
    (aminx, aminy, aminz), (amaxx, amaxy, amaxz) = a
    (bminx, bminy, bminz), (bmaxx, bmaxy, bmaxz) = b
    return (aminx <= bmaxx and amaxx >= bminx and
            aminy <= bmaxy and amaxy >= bminy and
            aminz <= bmaxz and amaxz >= bminz)

@dataclass
class Player:
    # position is center of the standing AABB
    x: float = 0.0
    y: float = 0.85     # half of height=1.7 so feet touch y=0
    z: float = 0.0
    yaw: float = 0.0    # degrees, turn left/right
    pitch: float = 0.0  # degrees, look up/down

    # collision shape
    radius: float = 0.3
    height: float = 1.7

    # movement
    speed: float = 3.0
    mouse_sens: float = 0.12   # yaw sensitivity
    mouse_sens_pitch: float = 0.12  # pitch sensitivity
    gravity: float = 0.0
    vel_y: float = 0.0
    grounded: bool = True

    def aabb(self) -> AABB:
        hw = self.radius
        hh = self.height * 0.5
        return ((self.x - hw, self.y - hh, self.z - hw),
                (self.x + hw, self.y + hh, self.z + hw))

def resolve_axis(player: "Player", colliders: List[AABB], axis: int, delta: float):
    """Move along one axis, clamp exactly to the first wall you hit.
       This prevents 'teleporting' and makes you stop flush against the border."""
    if abs(delta) < 1e-9:
        return

    # move tentatively
    if axis == 0:
        player.x += delta
    elif axis == 1:
        player.y += delta
    else:
        player.z += delta

    # Player half sizes (for clamping)
    half_w = player.radius
    half_h = player.height * 0.5
    EPS = 1e-4  # tiny gap to avoid re-overlap due to float error

    # check collisions and clamp to the blocking face on THIS axis
    while True:
        pmin, pmax = player.aabb()
        hit = False
        for (bmin, bmax) in colliders:
            # quick reject on other axes to avoid false hits
            if not aabb_overlap((pmin, pmax), (bmin, bmax)):
                continue

            # we overlapped; clamp on the axis we are resolving
            if axis == 0:  # X
                if delta > 0:
                    # moving +X: place player so pmax.x == bmin.x - EPS
                    player.x = bmin[0] - half_w - EPS
                else:
                    # moving -X: place player so pmin.x == bmax.x + EPS
                    player.x = bmax[0] + half_w + EPS
            elif axis == 1:  # Y
                if delta > 0:
                    player.y = bmin[1] - half_h - EPS
                else:
                    player.y = bmax[1] + half_h + EPS
                    player.grounded = True
                    player.vel_y = 0.0
            else:  # Z
                if delta > 0:
                    player.z = bmin[2] - half_w - EPS
                else:
                    player.z = bmax[2] + half_w + EPS

            hit = True
            break  # clamp to the first blocking box; then re-check

        if not hit:
            break


def move_player(
    player: Player,
    dt: float,
    input_fwd: float,
    input_right: float,
    mouse_dx: float,
    mouse_dy: float,
    colliders: List[AABB],
    lock_y: bool = True
):
    # ---- mouse look ----
    # invert dx sign so moving mouse right rotates view right (OpenGL RH convention)
    player.yaw   -= mouse_dx * player.mouse_sens
    player.pitch -= mouse_dy * player.mouse_sens_pitch
    # clamp pitch for FPS-style 360° yaw, ±89° pitch
    if player.pitch > 89.0: player.pitch = 89.0
    if player.pitch < -89.0: player.pitch = -89.0

    # ---- movement (XZ only; ignore pitch so you don't move up/down when looking up) ----
    yaw_rad = math.radians(player.yaw)
    fx, fz = math.sin(yaw_rad), math.cos(yaw_rad)        # forward on XZ
    rx, rz = -fz, fx                                     # right on XZ (RH system)

    vx = (fx * input_fwd + rx * input_right) * player.speed
    vz = (fz * input_fwd + rz * input_right) * player.speed

    vy = player.vel_y
    if player.gravity > 0.0 and not lock_y:
        vy -= player.gravity * dt
        player.grounded = False

    # axis-separated resolution => slide along walls
    resolve_axis(player, colliders, 0, vx * dt)  # X
    if not lock_y:
        resolve_axis(player, colliders, 1, vy * dt)  # Y
    resolve_axis(player, colliders, 2, vz * dt)  # Z

    if lock_y:
        player.y = player.height * 0.5  # keep eye height fixed

# ---------- helpers ----------
def make_wall(minx, miny, minz, maxx, maxy, maxz) -> AABB:
    return ((float(minx), float(miny), float(minz)),
            (float(maxx), float(maxy), float(maxz)))

# Basic debug drawer for AABBs (immediate mode)
# --- replace the old draw_debug_aabbs with this core-profile version ---
def draw_debug_aabbs(colliders, shader=None, color=(1.0, 0.0, 0.0)):
    """
    Core-profile safe debug drawer for AABBs.
    - colliders: list of ((minx,miny,minz),(maxx,maxy,maxz))
    - shader:    your bound Shader instance (optional but recommended).
                 If provided, we set uniforms: objectColor, useTexture=0, emissive=1
    - color:     RGB tuple for line color
    """
    import numpy as np
    from OpenGL.GL import (
        glGenVertexArrays, glBindVertexArray,
        glGenBuffers, glBindBuffer, glBufferData,
        glEnableVertexAttribArray, glVertexAttribPointer,
        glDrawArrays, glLineWidth,
        GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE,
        GL_LINES
    )

    # Lazy-create VAO/VBO once and stash on the function object
    if not hasattr(draw_debug_aabbs, "_vao"):
        draw_debug_aabbs._vao = glGenVertexArrays(1)
        draw_debug_aabbs._vbo = glGenBuffers(1)
        draw_debug_aabbs._cap = 0  # number of floats capacity

        glBindVertexArray(draw_debug_aabbs._vao)
        glBindBuffer(GL_ARRAY_BUFFER, draw_debug_aabbs._vbo)
        # position at location=0, 3 floats
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)

    # Build the line list (24 edges -> 24*2 vertices? actually 12 edges*2 verts = 24 verts per box)
    verts = []
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    for (mn, mx) in colliders:
        minx, miny, minz = mn
        maxx, maxy, maxz = mx
        c = [
            (minx,miny,minz),(maxx,miny,minz),(maxx,miny,maxz),(minx,miny,maxz),
            (minx,maxy,minz),(maxx,maxy,minz),(maxx,maxy,maxz),(minx,maxy,maxz)
        ]
        for a, b in edges:
            verts.extend(c[a]); verts.extend(c[b])

    if not verts:
        return

    arr = np.array(verts, dtype=np.float32)
    count = arr.size // 3  # number of vertices

    glBindVertexArray(draw_debug_aabbs._vao)
    glBindBuffer(GL_ARRAY_BUFFER, draw_debug_aabbs._vbo)

    # (re)allocate if needed
    if arr.size > draw_debug_aabbs._cap:
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_DYNAMIC_DRAW)
        draw_debug_aabbs._cap = arr.size
    else:
        # update within existing buffer
        from OpenGL.GL import glBufferSubData
        glBufferSubData(GL_ARRAY_BUFFER, 0, arr.nbytes, arr)

    # set uniforms if we got your Shader wrapper
    if shader is not None:
        try:
            shader.set_int('useTexture', 0)
        except Exception:
            pass
        try:
            shader.set_int('emissive', 1)
        except Exception:
            pass
        try:
            import numpy as _np
            shader.set_vec3('objectColor', _np.array(color, dtype=_np.float32))
        except Exception:
            pass

    glLineWidth(1.0)
    glDrawArrays(GL_LINES, 0, count)
