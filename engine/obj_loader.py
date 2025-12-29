# obj_loader.py — bulbs turn white when lights are ON (surface emissive only),
# plus we expose the material name per group so main.py can dim walls/floors.
import os, json, ctypes, math
import numpy as np
from OpenGL.GL import *
from .texture import load_texture_file

# ---------------- containers ----------------
class Group:
    __slots__ = ("vao", "vbo", "count", "color", "tex", "emissive", "emissive_color", "mtl")
    def __init__(self, vao, vbo, count, color, tex, emissive=False, emissive_color=(0.0,0.0,0.0), mtl=""):
        self.vao = vao
        self.vbo = vbo
        self.count = count
        self.color = color
        self.tex = tex
        self.emissive = emissive           # main.py toggles this with 'L'
        self.emissive_color = emissive_color
        self.mtl = mtl                      # <- expose material name

class Model:
    def __init__(self):
        self.groups = []
        self.lights = []  # (center, ke) REAL lights created from non-bulb emissive mats

# ---------------- small helpers ----------------
def _load_overrides(path):
    if path and os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _match_override(name, overrides):
    n = name.lower()
    for key, spec in overrides.items():
        if key.lower() in n:
            return spec or {}
    return {}

def _dominant_plane_from_normal(n):
    ax = np.abs(n)
    if ax[1] >= ax[0] and ax[1] >= ax[2]: return "xz"
    if ax[0] >= ax[2]: return "yz"
    return "xy"

def _project_point(p, plane):
    if plane == "xy": return np.array([p[0], p[1]], dtype=np.float32)
    if plane == "xz": return np.array([p[0], p[2]], dtype=np.float32)
    if plane == "yz": return np.array([p[1], p[2]], dtype=np.float32)
    return np.array([p[0], p[2]], dtype=np.float32)

def _apply_uv_transform(uv, spec, tex_size=None):
    u, v = float(uv[0]), float(uv[1])
    if spec.get("uv_flip_u"): u = 1.0 - u
    if spec.get("uv_flip_v"): v = 1.0 - v
    theta = math.radians(spec.get("uv_rotate_deg", 0.0))
    if abs(theta) > 1e-6:
        ct, st = math.cos(theta), math.sin(theta)
        u, v = (u * ct - v * st), (u * st + v * ct)
    su, sv = 1.0, 1.0
    if "uv_scale" in spec:
        su, sv = spec["uv_scale"]
    if spec.get("uv_keep_aspect") and tex_size and tex_size[0] and tex_size[1]:
        tw, th = float(tex_size[0]), float(tex_size[1])
        sv = su * (th / tw)
    ou, ov = spec.get("uv_offset", [0.0, 0.0])
    u = u * su + ou
    v = v * sv + ov
    return np.array([u, v], dtype=np.float32)

# ---------------- MTL loader ----------------
def _load_mtl(path):
    mats = {}
    cur = None
    if not path or not os.path.exists(path):
        return mats
    for line in open(path, "r", encoding="utf-8", errors="ignore"):
        s = line.strip()
        if not s or s.startswith("#"): continue
        t = s.split()
        k = t[0].lower()
        if k == "newmtl":
            cur = t[1]
            mats[cur] = {"Kd": (0.8, 0.8, 0.8), "Ke": (0.0, 0.0, 0.0), "map_Kd": None}
        elif k == "kd" and cur:
            try: mats[cur]["Kd"] = tuple(float(x) for x in t[1:4])
            except: pass
        elif k == "ke" and cur:
            try: mats[cur]["Ke"] = tuple(float(x) for x in t[1:4])
            except: pass
        elif k == "map_kd" and cur:
            rel = " ".join(t[1:]).strip().replace("\\", "/")
            mats[cur]["map_Kd"] = rel
    return mats

# ---------------- OBJ loader ----------------
def load_obj(obj_path, overrides_path=None):
    base_dir = os.path.dirname(obj_path)

    # find mtllib
    mtl_path = None
    for line in open(obj_path, "r", encoding="utf-8", errors="ignore"):
        if line.lower().startswith("mtllib "):
            mtl_path = os.path.join(base_dir, line.split(None, 1)[1].strip())
            print(f"Loading MTL for {obj_path}: {mtl_path}")
            break

    mats = _load_mtl(mtl_path)
    print(f"Materials for {obj_path}: {list(mats.keys())}")
    overrides = _load_overrides(overrides_path)

    v, vt, vn = [None], [None], [None]
    usemtl = "default"
    tris = []
    materials = set()

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"): continue
            t = s.split()
            op = t[0].lower()

            if op == "v" and len(t) >= 4:
                v.append(np.array([float(t[1]), float(t[2]), float(t[3])], dtype=np.float32))
            elif op == "vt" and len(t) >= 3:
                vt.append(np.array([float(t[1]), float(t[2])], dtype=np.float32))
            elif op == "vn" and len(t) >= 4:
                vn.append(np.array([float(t[1]), float(t[2]), float(t[3])], dtype=np.float32))
            elif op == "usemtl" and len(t) >= 2:
                usemtl = t[1]
                materials.add(usemtl)
            elif op == "f" and len(t) >= 4:
                ids = t[1:]

                def pidx(s):
                    p = s.split("/")
                    vi = int(p[0]) if p[0] else 0
                    ti = int(p[1]) if len(p) > 1 and p[1] else 0
                    ni = int(p[2]) if len(p) > 2 and p[2] else 0
                    return vi, ti, ni

                idxs = [pidx(x) for x in ids]
                for i in range(1, len(idxs) - 1):
                    tris.append((usemtl, idxs[0], idxs[i], idxs[i + 1]))

    # texture sizes (optional, for UV overrides)
    mat_tex_sizes = {}
    for mtl in materials:
        tex_path = None
        mkd = mats.get(mtl, {}).get("map_Kd")
        ov = _match_override(mtl, overrides)
        if mkd:
            tex_path = os.path.join(base_dir, mkd.replace("\\", "/"))
        elif "file" in ov:
            tex_path = os.path.join(base_dir, ov["file"])
        wh = None
        if tex_path and os.path.exists(tex_path):
            try:
                import pygame as pg
                surf = pg.image.load(tex_path)
                wh = surf.get_size()
            except Exception:
                wh = None
        mat_tex_sizes[mtl] = wh

    # group by material
    grouped = {}
    for (mtl, a, b, c) in tris:
        data = grouped.setdefault(mtl, {"verts": []})

        def fetch(idx):
            vi, ti, ni = idx
            p  = v[vi] if vi else np.zeros(3, dtype=np.float32)
            uv = vt[ti] if (ti and ti < len(vt)) else None
            n  = vn[ni] if (ni and ni < len(vn)) else None
            return p, uv, n

        p0, uv0, n0 = fetch(a)
        p1, uv1, n1 = fetch(b)
        p2, uv2, n2 = fetch(c)

        if n0 is None or n1 is None or n2 is None:
            nrm = np.cross(p1 - p0, p2 - p0)
            nrm = nrm / (np.linalg.norm(nrm) + 1e-8)
            n0 = n1 = n2 = nrm

        if uv0 is None or uv1 is None or uv2 is None:
            spec = _match_override(mtl, overrides)
            proj = (spec.get("uv_project", "auto") or "auto").lower()
            if proj == "auto":
                proj = _dominant_plane_from_normal(n0)
            uv0 = _project_point(p0, proj)
            uv1 = _project_point(p1, proj)
            uv2 = _project_point(p2, proj)
            tex_wh = mat_tex_sizes.get(mtl)
            uv0 = _apply_uv_transform(uv0, spec, tex_wh)
            uv1 = _apply_uv_transform(uv1, spec, tex_wh)
            uv2 = _apply_uv_transform(uv2, spec, tex_wh)

        data["verts"].extend([p0[0], p0[1], p0[2], n0[0], n0[1], n0[2], uv0[0], uv0[1]])
        data["verts"].extend([p1[0], p1[1], p1[2], n1[0], n1[1], n1[2], uv1[0], uv1[1]])
        data["verts"].extend([p2[0], p2[1], p2[2], n2[0], n2[1], n2[2], uv2[0], uv2[1]])

    # -------- build GL buffers & groups --------
    BULB_KEYWORDS       = ("bulb", "lamp", "led", "lightbulb", "filament", "incandescent")
    BULB_EMISSION_COLOR = np.array([1.0, 1.0, 0.95], dtype=np.float32)  # warm-white surface glow
    BULB_EMISSION_GAIN  = 1.0

    model = Model()

    for mtl, bundle in grouped.items():
        arr = np.array(bundle["verts"], dtype=np.float32)

        vbo = glGenBuffers(1)
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
        stride = 8 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # base color
        kd = mats.get(mtl, {}).get("Kd", (0.8, 0.8, 0.8))
        color = np.array(kd, dtype=np.float32)

        # load texture if any
        ov = _match_override(mtl, overrides) or {}
        uv_mode_repeat = ov.get("uv_mode", "repeat").lower() == "repeat"
        tex = None
        mkd = mats.get(mtl, {}).get("map_Kd")
        if mkd:
            tex, _ = load_texture_file(os.path.join(base_dir, mkd.replace("\\", "/")), mode_repeat=uv_mode_repeat)
        if tex is None and "file" in ov:
            tex, _ = load_texture_file(os.path.join(base_dir, ov["file"]), mode_repeat=uv_mode_repeat)

        # emissive flags
        name_l = mtl.lower()
        is_bulb = any(k in name_l for k in BULB_KEYWORDS)

        ke = mats.get(mtl, {}).get("Ke", (0.0, 0.0, 0.0))
        has_real_emission = any(channel > 0.0 for channel in ke)

        emissive = is_bulb or has_real_emission
        emissive_color = (BULB_EMISSION_COLOR * BULB_EMISSION_GAIN) if is_bulb else np.array(ke, dtype=np.float32)

        group = Group(vao, vbo, arr.size // 8, color, tex, emissive, emissive_color, mtl=mtl)
        model.groups.append(group)

        # REAL lights only for non-bulb emissive mats
        if has_real_emission and not is_bulb:
            positions = arr.reshape(-1, 8)[:, :3]
            center = np.mean(positions, axis=0)
            print(f"Emissive light source {mtl} @ {center} (Ke={ke})")
            model.lights.append((center, ke))

    return model
