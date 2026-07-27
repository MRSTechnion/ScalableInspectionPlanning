import os
import time
import pybullet as p
import pybullet_data
import math


# -----------------------------
# CONFIG
# -----------------------------
# OBJ_PATH = r"./OBJ/Bridge.obj"
OBJ_PATH = r"../../assets/OBJ/water_tower.obj"
# OBJ_PATH = r"../../assets/OBJ/MEA3.obj"
# OBJ_PATH = r"./OBJ/menara_kl.obj"
# OBJ_PATH = r"./OBJ/tower_bridge_obj/tower_bridge.obj"


USE_CONCAVE_COLLISION = True                     # useful for environment/static meshes
IS_STATIC = True                                 # static object = mass 0
ORIENTATION = "vertical_x"  # "horizontal", "vertical_x", or "vertical_y"

ORIENTATIONS = {
    "horizontal": [0, 0, 0],
    "vertical_x": [math.pi / 2, 0, 0],
    "vertical_y": [0, math.pi / 2, 0],
}

START_EULER = ORIENTATIONS[ORIENTATION]
MESH_SCALE = [0.05, 0.05, 0.05]
# MESH_SCALE = [0.0005, 0.0005, 0.0005]
# MESH_SCALE = [1, 1, 1]
START_POS = [0, 0, 0]

def main():
    if not os.path.exists(OBJ_PATH):
        raise FileNotFoundError(f"Missing OBJ: {OBJ_PATH}")

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 1]
    )
    
    p.loadURDF("plane.urdf")

    quat = p.getQuaternionFromEuler(START_EULER)

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=OBJ_PATH,
        meshScale=MESH_SCALE,
        specularColor=[0.5, 0.5, 0.5]
    )

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=OBJ_PATH,
        meshScale=MESH_SCALE,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=START_POS,
        baseOrientation=quat
    )

    print("body id:", body)
    print("visual shape id:", visual_shape)
    print("collision shape id:", collision_shape)

    aabb_min, aabb_max = p.getAABB(body)
    print("AABB min:", aabb_min)
    print("AABB max:", aabb_max)

    center = [(aabb_min[i] + aabb_max[i]) / 2 for i in range(3)]
    size = [aabb_max[i] - aabb_min[i] for i in range(3)]
    print("AABB center:", center)
    print("AABB size:", size)

    # draw bounding box
    corners = [
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i, j in edges:
        p.addUserDebugLine(corners[i], corners[j], [0, 1, 0], 2)

    # mark origin
    p.addUserDebugLine([0, 0, 0], [0.5, 0, 0], [1, 0, 0], 3)
    p.addUserDebugLine([0, 0, 0], [0, 0.5, 0], [0, 1, 0], 3)
    p.addUserDebugLine([0, 0, 0], [0, 0, 0.5], [0, 0, 1], 3)

    # center camera on object
    p.resetDebugVisualizerCamera(
        cameraDistance=max(1.0, max(size) * 2),
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=center
    )

    while p.isConnected():
        p.stepSimulation()
        time.sleep(1 / 240)

if __name__ == "__main__":
    main()