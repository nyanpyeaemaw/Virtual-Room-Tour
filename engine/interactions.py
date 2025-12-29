# engine/interactions.py
import numpy as np
from OpenGL.GL import *
from .obj_loader import load_obj
import numpy as np
from dataclasses import dataclass
from engine.character_controller import draw_debug_aabbs
# -------- screen->world ray --------
def _inv(m): return np.linalg.inv(m)

def _unproject(nx, ny, nz, inv_vp):
    p = np.array([nx, ny, nz, 1.0], dtype=np.float32)
    w = inv_vp @ p
    return (w[:3] / (w[3] if abs(w[3]) > 1e-8 else 1.0)).astype(np.float32)



def ray_hits_aabb(ray_origin, ray_dir, aabb):
    tmin = -float('inf')
    tmax = float('inf')
    for i in range(3):
        if abs(ray_dir[i]) < 1e-8:
            if ray_origin[i] < aabb[0][i] or ray_origin[i] > aabb[1][i]:
                return False
        else:
            t1 = (aabb[0][i] - ray_origin[i]) / ray_dir[i]
            t2 = (aabb[1][i] - ray_origin[i]) / ray_dir[i]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
            if tmin > tmax:
                return False
    return True

def make_mouse_ray(mx, my, screen_size, view, proj):
    import numpy as np
    w, h = screen_size
    my = (h - 1) - my  # flip once here; don’t flip elsewhere

    x = (2.0 * mx / max(w,1)) - 1.0
    y = (2.0 * my / max(h,1)) - 1.0

    invP = np.linalg.inv(proj)
    invV = np.linalg.inv(view)

    ray_clip = np.array([x, y, -1.0, 1.0], np.float32)
    ray_eye  = invP @ ray_clip
    ray_eye  = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0], np.float32)

    ray_world4 = invV @ ray_eye
    rd = ray_world4[:3]
    rd = rd / (np.linalg.norm(rd) + 1e-8)

    ro = (invV @ np.array([0,0,0,1], np.float32))[:3]
    return ro, rd


def ray_aabb(ro, rd, mn, mx):
    tmin, tmax = -1e30, 1e30
    for i in range(3):
        if abs(rd[i]) < 1e-12:
            if ro[i] < mn[i] or ro[i] > mx[i]:
                return None
        else:
            t1 = (mn[i] - ro[i]) / rd[i]
            t2 = (mx[i] - ro[i]) / rd[i]
            if t1 > t2: t1, t2 = t2, t1
            tmin = max(tmin, t1); tmax = min(tmax, t2)
            if tmin > tmax: return None
    return tmin if tmax >= 0 else None

# -------- tiny math helpers (match your main.py style) --------
def translate(tx, ty, tz):
    M = np.eye(4, dtype=np.float32); M[:3, 3] = [tx, ty, tz]; return M
def rotate_y(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    M = np.eye(4, dtype=np.float32); M[0,0]=c; M[0,2]=s; M[2,0]=-s; M[2,2]=c; return M
def scale(sx, sy, sz):
    M = np.eye(4, dtype=np.float32); M[0,0]=sx; M[1,1]=sy; M[2,2]=sz; return M

def hinge_y(angle_deg, pivot_world):
    px, py, pz = pivot_world
    return translate(px, py, pz) @ rotate_y(angle_deg) @ translate(-px, -py, -pz)

# -------- door actor --------
class DoorActor:
    def __init__(self, obj_path, base_trs, hinge_world, pick_aabb, open_angle=90.0):
        self.model = load_obj(obj_path)
        self.base_trs = base_trs
        self.hinge = np.array(hinge_world, dtype=np.float32)
        self.aabb = (np.array(pick_aabb[0], dtype=np.float32),
                     np.array(pick_aabb[1], dtype=np.float32))
        self.open_angle = float(open_angle)
        self.is_open = False
        self._angle = 0.0   # animated angle
        self.hovered = False

        print("[door] loaded", len(self.model.groups), "groups")


    def update(self, dt):
        target = self.open_angle if self.is_open else 0.0
        # critically damped-ish step
        max_speed = 220.0  # deg/s
        step = np.clip(target - self._angle, -max_speed*dt, max_speed*dt)
        self._angle += step

    def set_open(self, v: bool):
        self.is_open = bool(v)

    def toggle(self):
        self.is_open = not self.is_open

    # def test_hover(self, mx, my, viewport_wh, view, proj):
    #     ro, rd = make_mouse_ray(mx, my, viewport_wh, view, proj)
    #     self.hovered = ray_aabb(ro, rd, *self.aabb) is not None
    #     return self.hovered

    def test_hover(self, mx, my, viewport_wh, view, proj):
        # Always cast from the center of the screen (crosshair)
        # w, h = viewport_wh
        # cx, cy = w // 2, h // 2
        # ro, rd = make_mouse_ray(cx, cy, viewport_wh, view, proj)
        ro, rd = make_mouse_ray(mx, my, viewport_wh, view, proj)

        self.hovered = ray_aabb(ro, rd, *self.aabb) is not None
        return self.hovered


    def model_matrix(self):
        return hinge_y(self._angle, self.hinge) @ self.base_trs
    
    
    def draw(self, shader):
        M = self.model_matrix()
        shader.set_mat4('model', M)
        draw_debug_aabbs([

            ((self.hinge[0] - 0.02, self.hinge[1],     self.hinge[2] - 0.02),
            (self.hinge[0] + 0.02, self.hinge[1] + 2, self.hinge[2] + 0.02))
        ],     

            shader=shader, color=(1.0, 0.0, 1.0))  # Purple hinge marker
        # debug
        if not hasattr(self, "_printed_pose"):
            print("[door] model Txyz this frame =",
              float(M[0,3]), float(M[1,3]), float(M[2,3]))
        self._printed_pose = True

        for g in self.model.groups:
            shader.set_int('emissive', 1 if g.emissive else 0)
            # shader.set_vec3('objectColor', g.color)
            shader.set_vec3('objectColor', np.array([0.0, 0.0, 0.0], dtype=np.float32))  

            if g.tex:
                shader.set_int('useTexture', 1)
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, g.tex)
                shader.set_int('tex0', 0)
            else:
                shader.set_int('useTexture', 0)
            glBindVertexArray(g.vao); glDrawArrays(GL_TRIANGLES, 0, g.count)
    def toggle_from_position(self, _=None):
        self.open_angle = +90  # always open into kitchen
        self.toggle()





# -------- lamp actor --------
class LampActor:
    def __init__(self, pick_aabb, pos, intensity=1.6):
        self.aabb = (np.array(pick_aabb[0], dtype=np.float32),
                     np.array(pick_aabb[1], dtype=np.float32))
        self.pos = np.array(pos, dtype=np.float32)
        self.intensity = intensity
        self.on = False
        self.hovered = False

    def toggle(self):
        self.on = not self.on

    def test_hover(self, mx, my, viewport_wh, view, proj):
        # w, h = viewport_wh
        # cx, cy = w // 2, h // 2
        # ro, rd = make_mouse_ray(cx, cy, viewport_wh, view, proj)
        ro, rd = make_mouse_ray(mx, my, viewport_wh, view, proj)
        # print("[hover] Ray origin:", ro, "Ray dir:", rd)
        # print("[hover] AABB min:", self.aabb[0], "max:", self.aabb[1])
        self.hovered = ray_aabb(ro, rd, *self.aabb) is not None
        # print("[hover] result:", self.hovered)

        return self.hovered
    
def ray_sphere(ro, rd, center, radius):
    """Ray-sphere hit test. Returns t or None."""
    oc = ro - center
    b = np.dot(oc, rd)         # note: rd is normalized
    c = np.dot(oc, oc) - radius*radius
    disc = b*b - c
    if disc < 0.0:
        return None
    t = -b - np.sqrt(disc)
    return t if t >= 0.0 else None


class WaterActor:
    def __init__(self, pick_aabb, flow_pos):
        self.aabb = (np.array(pick_aabb[0], dtype=np.float32),
                     np.array(pick_aabb[1], dtype=np.float32))
        self.pos = np.array(flow_pos, dtype=np.float32)
        self.on = False
        self.hovered = False

    def toggle(self):
        self.on = not self.on

    # def test_hover(self, mx, my, viewport_wh, view, proj):
    #     # Always use screen center
    #     w, h = viewport_wh
    #     cx, cy = w // 2, h // 2
    #     ro, rd = make_mouse_ray(cx, cy, viewport_wh, view, proj)

    #     print("[water] Ray origin:", ro)
    #     print("[water] Ray dir:", rd)
    #     print("[water] AABB min:", self.aabb[0], "max:", self.aabb[1])
        
    #     hit = ray_aabb(ro, rd, *self.aabb) is not None
    #     print("[water] hover result:", hit)
    #     self.hovered = hit
    #     return hit

    def test_hover(self, mx, my, viewport_wh, view, proj):
        ro, rd = make_mouse_ray(mx, my, viewport_wh, view, proj)

        # center of the AABB being tested (world space)
        center = 0.5*(self.aabb[0] + self.aabb[1])
        to_center = center - ro
        dist = np.linalg.norm(to_center)
        cosang = float(np.dot(rd, to_center/(dist+1e-8)))

        # print("[water DEBUG] ray_origin:", ro, "ray_dir:", rd)
        # print("[water DEBUG] aabb_center:", center, "dist:", dist, "cos(angle):", cosang)


        hit_box = ray_aabb(ro, rd, self.aabb[0], self.aabb[1]) is not None
        hit_sphere = ray_sphere(ro, rd, self.pos, radius=0.18) is not None
        self.hovered = bool(hit_box or hit_sphere)
        return self.hovered

    def draw(self, shader):
        if not self.on:
            return

        # Draw the "water stream" as a tall thin box
        flow_box = (
            self.pos + np.array([-0.03, -1.0, -0.03], dtype=np.float32),
            self.pos + np.array([+0.03,  0.0,  +0.03], dtype=np.float32)
        )
        draw_debug_aabbs([flow_box], shader=shader, color=(0.3, 0.6, 1.0))  # light blue

# ----------------------------Swtich-----------------------------------------------

class SwitchActor:
    def __init__(self, aabb, callback):
        self.aabb = aabb
        self.hovered = False
        self.on = False
        self.callback = callback

    def toggle(self):
        self.on = not self.on
        self.callback(self.on)

    # def test_hover(self, mx, my, screen_size, view, proj):
    #     ray_origin, ray_dir = make_mouse_ray(mx, my, screen_size, view, proj)
    #     self.hovered = ray_hits_aabb(ray_origin, ray_dir, self.aabb)
    def test_hover(self, mx, my, screen_size, view, proj):
        ray_origin, ray_dir = make_mouse_ray(mx, my, screen_size, view, proj)
        hit = ray_hits_aabb(ray_origin, ray_dir, self.aabb)
        if hit:
            center = 0.5 * (np.array(self.aabb[0]) + np.array(self.aabb[1]))
            dist = np.linalg.norm(center - ray_origin)
            self.hovered = dist < 1.0  # allow toggle only if within 1m
        else:
            self.hovered = False
        return self.hovered




    def draw(self, shader):

        pass

class DebugSwitchBox:
    def __init__(self, aabb):
        self.aabb = [list(aabb[0]), list(aabb[1])]
        self.speed = 0.01

    def move(self, dx=0, dy=0, dz=0):
        for i in range(3):
            self.aabb[0][i] += [dx, dy, dz][i]
            self.aabb[1][i] += [dx, dy, dz][i]

    def get(self):
        return (tuple(self.aabb[0]), tuple(self.aabb[1]))




