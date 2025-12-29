# main.py — per-room lights, hard OFF toggle, kitchen surface dimmer
import os, json
import numpy as np
import pygame as pg
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *

from engine.shader import Shader
from engine.obj_loader import load_obj
from engine.character_controller import Player, move_player, draw_debug_aabbs  
from engine.interactions import DoorActor, translate as T, rotate_y as RY, scale as S
from engine.interactions import LampActor, SwitchActor
from engine.character_controller import draw_debug_aabbs
from engine.interactions import make_mouse_ray
from engine.interactions import WaterActor

kitchen_bulbs_on = {"value": True} 
living_bulbs_on = {"value": True}
# ---------- math helpers ----------
def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect; M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar); M[2, 3] = (2 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1
    return M

def lookAt(eye, center, up):
    def normalize(v): n = np.linalg.norm(v); return v if n == 0 else v / n
    f = normalize(center - eye); s = normalize(np.cross(f, up)); u = np.cross(s, f)
    M = np.identity(4, dtype=np.float32); M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
    T = np.identity(4, dtype=np.float32); T[:3, 3] = -eye
    return M @ T

def translate(tx, ty, tz):
    M = np.identity(4, dtype=np.float32); M[:3, 3] = [tx, ty, tz]; return M
def rotate_x(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    M = np.identity(4, dtype=np.float32); M[1, 1] = c; M[1, 2] = -s; M[2, 1] = s; M[2, 2] = c; return M
def rotate_y(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    M = np.identity(4, dtype=np.float32); M[0, 0] = c; M[0, 2] = s; M[2, 0] = -s; M[2, 2] = c; return M
def rotate_z(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    M = np.identity(4, dtype=np.float32); M[0, 0] = c; M[0, 1] = -s; M[1, 0] = s; M[1, 1] = c; return M
def scale(sx, sy, sz):
    M = np.identity(4, dtype=np.float32); M[0, 0] = sx; M[1, 1] = sy; M[2, 2] = sz; return M

# ---------- collider helpers ----------
def aabb_after_model(min_local, max_local, M):
    l = np.array(min_local, dtype=np.float32)
    u = np.array(max_local, dtype=np.float32)
    corners = np.array([
        [l[0], l[1], l[2], 1], [u[0], l[1], l[2], 1],
        [u[0], u[1], l[2], 1], [l[0], u[1], l[2], 1],
        [l[0], l[1], u[2], 1], [u[0], l[1], u[2], 1],
        [u[0], u[1], u[2], 1], [l[0], u[1], u[2], 1]
    ], dtype=np.float32)
    world = (M @ corners.T).T[:, :3]
    return (tuple(world.min(axis=0).tolist()), tuple(world.max(axis=0).tolist()))

def load_boxes_json(path):
    if not os.path.exists(path): return []
    data = json.load(open(path, "r", encoding="utf-8"))
    return data.get("boxes", [])

def rebuild_world_colliders_from_boxes(boxes, mats):
    out = []
    for b in boxes:
        if b.get("solid", True) is False: continue
        M = mats.get(b.get("space", "living")); 
        if M is None: continue
        out.append(aabb_after_model(b["min"], b["max"], M))
    return out

# ---------- per-room light upload ----------
def push_lights(shader, lights, src_name, light):
    """Upload only lights for the given room. Hard OFF = zero lights."""
    if not light:
        shader.set_int('numLights', 0)
        shader.set_vec3_array('lightPos', np.zeros((0,3), dtype=np.float32))
        shader.set_vec3_array('lightColor', [])
        shader.set_float_array('lightIntensity', [])
        return
    positions, colors, intensities = [], [], []
    for src, pos, col, inten in lights:
        if src == src_name:
            positions.append(pos); colors.append(col); intensities.append(float(inten))
    shader.set_int('numLights', len(positions))
    if positions:
        shader.set_vec3_array('lightPos', np.array(positions, dtype=np.float32))
        shader.set_vec3_array('lightColor', colors)
        shader.set_float_array('lightIntensity', intensities)
    else:
        shader.set_vec3_array('lightPos', np.zeros((0,3), dtype=np.float32))
        shader.set_vec3_array('lightColor', [])
        shader.set_float_array('lightIntensity', [])

# ---------- per-surface dimming ----------
DIM_KEYWORDS = ("wall", "walls", "floor", "tile", "ceiling")
def gain_for_group(group, model_name, cfg):
    # dim only kitchen walls/floors; emissive groups are handled in draw loop
    if model_name == "kitchen":
        name = (getattr(group, "mtl", "") or "").lower()
        if any(k in name for k in DIM_KEYWORDS):
            return float(cfg.get("kitchen_surface_gain", 0.6))
    return 1.0
# ------Objects Interaction--------------
def distance_to_aabb_screen(mx, my, screen_size, view, proj, aabb):
    center = 0.5 * (np.array(aabb[0]) + np.array(aabb[1]))
    center4 = np.append(center, 1.0)
    clip = proj @ view @ center4
    ndc = clip[:3] / clip[3]
    screen_x = (ndc[0] * 0.5 + 0.5) * screen_size[0]
    screen_y = (1.0 - (ndc[1] * 0.5 + 0.5)) * screen_size[1]
    dx = mx - screen_x
    dy = my - screen_y
    return dx*dx + dy*dy  # square distance (faster)

# for water pump -> not working
def transform_aabb(aabb, mat):
    """
    Transforms an AABB (min, max) using the given 4x4 transformation matrix.
    Returns a new world-space AABB (min, max).
    """
    min_pt = np.array([*aabb[0], 1.0], dtype=np.float32)
    max_pt = np.array([*aabb[1], 1.0], dtype=np.float32)

    # Build all 8 corners
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2], 1.0],
        [max_pt[0], min_pt[1], min_pt[2], 1.0],
        [min_pt[0], max_pt[1], min_pt[2], 1.0],
        [max_pt[0], max_pt[1], min_pt[2], 1.0],
        [min_pt[0], min_pt[1], max_pt[2], 1.0],
        [max_pt[0], min_pt[1], max_pt[2], 1.0],
        [min_pt[0], max_pt[1], max_pt[2], 1.0],
        [max_pt[0], max_pt[1], max_pt[2], 1.0],
    ])

    transformed = (mat @ corners.T).T[:, :3]
    min_out = transformed.min(axis=0)
    max_out = transformed.max(axis=0)
    return (min_out, max_out)


# for kitch light bulbs
def toggle_kitchen_bulbs(state):
    kitchen_bulbs_on["value"] = state
    print("Kitchen bulbs are", "ON" if state else "OFF")
    

def toggle_living_bulbs(state):
    living_bulbs_on["value"] = state
    print("Living bulbs are", "ON" if state else "OFF")
# ---------- main ----------
def main():
    cfg = json.load(open('assets/CONFIG.json', 'r', encoding='utf-8'))

    pg.init()
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
    screen = pg.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
    glViewport(0, 0, *screen.get_size())

    cam_start = np.array(cfg.get('camera_start', [0.0, 1.6, 3.0]), dtype=np.float32)
    player = Player(x=float(cam_start[0]), y=1.7 * 0.5, z=float(cam_start[2]),
                    speed=3.0, radius=0.25, height=1.7)
    # spawn near living room like your project
    player.x, player.z = 6.05, 0.85 + 1.0

    pg.mouse.set_visible(False); pg.event.set_grab(True); pg.mouse.get_rel()

    shader = Shader('shaders/basic.vert', 'shaders/basic.frag')
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.07, 0.08, 0.1, 1.0)

    # Models
    kitchen = load_obj("assets/Modern Kitchen.obj")
    living  = load_obj("assets/InteriorWithNewSofa.obj")
    doorPath = "assets/door.obj"

    # Transforms
    kitchen_mat = translate(0.0, 0.0, 0.0) @ rotate_z(0.0) @ rotate_y(0.0) @ rotate_x(0.0) @ scale(1.0,1.0,1.0)
    living_mat  = translate(6.050000190734863, 0.0, 0.8499999046325684) @ rotate_z(180.0) @ rotate_y(0.0) @ rotate_x(180.0) @ scale(0.88,0.88,0.8)
    
    #-------Interaction-> door-------------------------------------------------------
    door_base = T(3.570, 0.0, -0.85) @ RY(-90) @ S(1.31, 0.91, 1)    
    door_hinge = (3.570, 0.0, -0.24) 
    door_aabb = ((3.50, 0.0, -2.2), (4.00, 2.00, 0.3)) 
    door = DoorActor(doorPath, door_base, door_hinge, door_aabb, open_angle=90.0)
    mx, my = 0, 0

    # ------Interaction -> water pump-----------------------------------
    water_aabb_local = ((-3.35, 0.60, -3.85), (-2.85, 1.40, -3.35))
    faucet_local_pos = np.array([-3.09, 1.02, -3.60], dtype=np.float32)
    water_aabb_world = transform_aabb(water_aabb_local, kitchen_mat)
    faucet_world_pos = (kitchen_mat @ np.array([*faucet_local_pos, 1.0], np.float32))[:3]
    water = WaterActor(water_aabb_world, faucet_world_pos)


    # ------Interaction -> light switch (kitchen)--------------


    switch_aabb = ((0.5, 1.0, 0.3), (1.0, 2.0, 0.8))
    switch = SwitchActor(switch_aabb, toggle_kitchen_bulbs)

    # -----Interaction -> lamp (living room)
    lamp_world_aabb = ((4.0, 0.0, 1.0), (5.0, 2.0, 2.0))
    lx = 0.5 * (lamp_world_aabb[0][0] + lamp_world_aabb[1][0])
    lz = 0.5 * (lamp_world_aabb[0][2] + lamp_world_aabb[1][2])
    ly = lamp_world_aabb[1][1] * 0.95
    lamp_pos = (lx, ly, lz)
    lamp = LampActor(lamp_world_aabb,lamp_pos)

    # Colliders
    mats = {"living": living_mat, "kitchen": kitchen_mat}
    boxes_local = load_boxes_json("assets/colliders.json")
    world_colliders = rebuild_world_colliders_from_boxes(boxes_local, mats)

    # Lighting state (start ON; press L to toggle hard OFF)
    lamp_on = True
    ambient_base = float(cfg.get('ambient_base', 0.0))

    # Optional: allow starting OFF via config
    lamp_on = bool(cfg.get('lights_start_on', True))

    clock = pg.time.Clock()
    editor_mode = False
    selected = -1
    running = True

    while running:
        dt = clock.tick(120) / 1000.0

        for e in pg.event.get():
            if e.type == pg.QUIT: running = False
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE: running = False
                if e.key == pg.K_l:  
                    lamp_on = not lamp_on

                if e.key == pg.K_o:
                    if switch.hovered:
                        switch.toggle()

                if e.key == pg.K_e:
                    editor_mode = not editor_mode
                    selected = 0 if editor_mode and boxes_local else -1
                if e.key == pg.K_i:
                    hovered = []

                    if door.hovered:
                        hovered.append(("door", door, door.aabb))
                    if lamp.hovered:
                        hovered.append(("lamp", lamp, lamp.aabb))
                    if water.hovered:
                        hovered.append(("water", water, water.aabb))


                    if hovered:
                        hovered.sort(key=lambda item: distance_to_aabb_screen(mx, my, screen.get_size(), view, proj, item[2]))
                        top = hovered[0][1]
                        top.toggle()
                        print(f"{hovered[0][0].capitalize()} toggled via [I] key")

                        if hovered[0][0] == "lamp":
                            living_bulbs_on["value"] = not living_bulbs_on["value"]
                        if hovered[0][0] == "switch":
                            kitchen_bulbs_on["value"] = not kitchen_bulbs_on["value"]



            # -------- Collect lights (emissive + your manual kitchen bulbs) --------
        all_lights = []  # (src_name, pos, color, intensity)
        base_intensity     = float(cfg.get('lamp_intensity', 1.0))
        kitchen_mult       = float(cfg.get('kitchen_bulb_intensity', 0.1))
        living_mult        = float(cfg.get('living_bulb_intensity', 1.0))
        warm_white         = np.array([1.0, 0.96, 0.85], dtype=np.float32)
        


        # (1) emissive lights discovered from models
        for model, mat, model_name in [(kitchen, kitchen_mat, "kitchen"), (living, living_mat, "living")]:
            for center, ke in model.lights:
                world_center = (mat @ np.append(center, 1.0))[:3]
                intensity = float(np.max(ke)) * base_intensity
                intensity *= (kitchen_mult if model_name == "kitchen" else living_mult)
                color = (np.array(ke, dtype=np.float32) / max(intensity, 1e-6)) if intensity > 0 else np.array([1.0,1.0,1.0], dtype=np.float32)
                all_lights.append((model_name, world_center.astype(np.float32), color.astype(np.float32), intensity))


        # (2) your measured kitchen bulbs
        kitchen_bulbs = [
            (-0.000, 2.400,  1.400),
            (-1.600, 2.250,  1.400),
            ( 1.600, 2.350,  1.300),
            ( 0.000, 2.500, -1.350),
            ( 1.700, 2.200, -1.450),
            (-1.650, 2.450, -1.400)
            # ( 0.000, 1.900,  0.000),
            # (-0.700, 1.900, -0.050),
            # ( 0.650, 2.000, -0.050),
        ]

        # (3) manual living-room bulbs (approximate ceiling points)
        # Center is your LIVING_TRANSLATION; ceiling ~2.3m. Adjust if needed.
        # Manual living-room bulbs (approx ceiling at y ≈ 2.3)
        living_bulbs = [
            (6.05, 2.30, 0.85),   # center
            (6.05 - 1.20, 2.30, 0.85 - 0.60),   # left/back
            (6.05 + 1.20, 2.30, 0.85 + 0.60),   # right/front
        ]

        if kitchen_bulbs_on["value"]:
            for pos in kitchen_bulbs:
                all_lights.append(("kitchen", np.array(pos, dtype=np.float32), warm_white, intensity))

        if living_bulbs_on["value"]:
            for pos in living_bulbs:
                all_lights.append(("living", np.array(pos, dtype=np.float32), warm_white, base_intensity * living_mult))

        # Fallback if nothing found
        if not all_lights:
            fallback_positions = np.array(cfg.get('lamp_positions', []), dtype=np.float32)
            default_color = np.array([1.0, 0.95, 0.85], dtype=np.float32)
            for pos in fallback_positions:
                all_lights.append(("kitchen", pos.astype(np.float32), default_color, base_intensity))
                all_lights.append(("living",  pos.astype(np.float32), default_color, base_intensity))

                
        keys = pg.key.get_pressed()
        input_fwd  = float(keys[pg.K_w]) - float(keys[pg.K_s])
        input_right= float(keys[pg.K_d]) - float(keys[pg.K_a])
        player.speed = 5.0 if (keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]) else 3.0
        mouse_dx, mouse_dy = pg.mouse.get_rel()
        move_player(player, dt, input_fwd, input_right, mouse_dx, mouse_dy, world_colliders, lock_y=True)


        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        w, h = pg.display.get_surface().get_size()

        yaw = np.radians(player.yaw); pitch = np.radians(player.pitch)
        front = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)], dtype=np.float32)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(front, world_up); right /= max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, front)
        cam_pos = np.array([player.x, player.y + player.height * 0.5 - 0.1, player.z], dtype=np.float32)
        
        glViewport(0, 0, w, h)
        view = lookAt(cam_pos, cam_pos + front, up)
        proj = perspective(60.0, max(w / h, 1e-3), 0.1, 200.0)

        mx, my = pg.mouse.get_pos()
        lamp.test_hover(mx, my, (w, h), view, proj)
        door.test_hover(mx, my, (w, h), view, proj)
        water.test_hover(mx, my, (w, h), view, proj)
        # water.hovered = True
        switch.test_hover(mx, my, (w, h), view, proj)

        shader.use()
        shader.set_int('sRGBTextures', 1)          # decode textures from sRGB (looks more natural)
        shader.set_float('exposure', 1.35)          # raise to 1.2–1.5 if scene looks dark after ACES
        # Optional: control per-light reach in shader; start around 8.0–10.0
        shader.set_float('lightRange', 10.0)

        shader.set_mat4('view', view)
        shader.set_mat4('projection', proj)
        shader.set_vec3('viewPos', cam_pos)
        shader.set_float(
        'ambientBase',
        (ambient_base if lamp_on else float(cfg.get('ambient_off_when_lamps_off', 0.03))))


        # --- KITCHEN ---
        shader.set_mat4('model', kitchen_mat)
        push_lights(shader, all_lights, "kitchen", kitchen_bulbs_on["value"])
        for g in kitchen.groups:
            # emissive strictly follows lamp_on
            shader.set_int('emissive', 1 if g.emissive and kitchen_bulbs_on["value"] else 0)
            shader.set_vec3('emissionColor', g.emissive_color if g.emissive and kitchen_bulbs_on["value"] else np.array([0.0,0.0,0.0], dtype=np.float32))
            # dim only non-emissive kitchen surfaces
            shader.set_float('surfaceGain', 1.0 if (g.emissive and kitchen_bulbs_on["value"]) else gain_for_group(g, "kitchen", cfg))
            shader.set_vec3('objectColor', g.color)
            if g.tex:
                shader.set_int('useTexture', 1)
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, g.tex); shader.set_int('tex0', 0)
            else:
                shader.set_int('useTexture', 0)
            glBindVertexArray(g.vao); glDrawArrays(GL_TRIANGLES, 0, g.count); glBindVertexArray(0)

        # --- LIVING ---
        shader.set_mat4('model', living_mat)
        push_lights(shader, all_lights, "living",living_bulbs_on["value"])
        for g in living.groups:
            shader.set_int('emissive', 1 if g.emissive and kitchen_bulbs_on["value"] else 0)
            shader.set_vec3('emissionColor', g.emissive_color if g.emissive and lamp_on else np.array([0.0,0.0,0.0], dtype=np.float32))
            shader.set_float('surfaceGain', 1.0)  # living room unaffected
            shader.set_vec3('objectColor', g.color)
            if g.tex:
                shader.set_int('useTexture', 1)
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, g.tex); shader.set_int('tex0', 0)
            else:
                shader.set_int('useTexture', 0)
            glBindVertexArray(g.vao); glDrawArrays(GL_TRIANGLES, 0, g.count); glBindVertexArray(0)
        
        # if kitchen_bulbs_on["value"]:  # Only add if turned ON
        #     for pos in kitchen_bulbs:
        #         all_lights.append(("kitchen", np.array(pos, dtype=np.float32), warm_white, intensity))

        # # Living bulbs — only if turned ON
        # if living_bulbs_on["value"]:
        #     for pos in living_bulbs:
        #         all_lights.append(("living", np.array(pos, dtype=np.float32), warm_white, base_intensity * living_mult))
        door.update(dt)
        door.draw(shader)
        water.draw(shader)
        switch.draw(shader)

        # --- Debug colliders ---
        if editor_mode and world_colliders:
            I = np.eye(4, dtype=np.float32)
            shader.set_mat4('model', I)
            glDisable(GL_DEPTH_TEST)
            draw_debug_aabbs(world_colliders, shader=shader, color=(1.0, 0.0, 0.0))
            # (optional selected box draw omitted for brevity)
            glEnable(GL_DEPTH_TEST)

        pg.display.set_caption(f'Kitchen + Living  |  Lights: {"ON" if lamp_on else "OFF"}')
        pg.display.flip()

    pg.quit()

if __name__ == '__main__':
    main()


#   //     {
#     //   "name": "lamp",
#     //   "space": "living",
#     //   "min": [ -0.15, 0.0, -0.15 ],
#     //   "max": [  0.15, 1.8,  0.15 ],
#     //   "solid": false
#     // },