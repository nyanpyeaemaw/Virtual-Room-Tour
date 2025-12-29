# colliders.py
# Define your solid objects (walls, counters, closed doors) as AABBs.
# Units should match your OBJ scene. Start with a perimeter and add big obstacles.

from engine.character_controller import make_wall

BASE_X, BASE_Z = 6.05, 0.85  # match LIVING_TRANSLATION in main.py

def build_world_colliders():
    colliders = []
    H = 3.0
    T = 0.2

    # perimeter WRT living room placement (shifted by BASE_X/Z)
    def wall(minx, miny, minz, maxx, maxy, maxz):
        return make_wall(minx+BASE_X, miny, minz+BASE_Z, maxx+BASE_X, maxy, maxz+BASE_Z)

    # Example 10x10 room around (BASE_X, BASE_Z)
    colliders.append(wall(-5.0, 0.0,  5.0 - T,  5.0, H,  5.0))   # north
    colliders.append(wall(-5.0, 0.0, -5.0,      5.0, H, -5.0 + T))# south
    colliders.append(wall(-5.0, 0.0, -5.0,     -5.0 + T, H,  5.0))# west
    colliders.append(wall( 5.0 - T, 0.0, -5.0,  5.0,     H,  5.0))# east

    # leave a door gap by not adding a collider section where the door is supposed to be

    # obstacle (e.g., island), also shifted
    colliders.append(wall(-1.2, 0.0, -0.6, 1.2, 1.0, 0.6))

    return colliders

