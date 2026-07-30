import pybullet as p


class Tray:
    def __init__(self, base_pos, glob_scale):
        self.base_position = base_pos
        self.global_scaling = glob_scale
        self.id = p.loadURDF(
            "tray/traybox.urdf",
            basePosition=base_pos,
            useFixedBase=True,
            globalScaling=glob_scale,
        )
        aabb_min, aabb_max = p.getAABB(self.id)
        self.half_length = (aabb_max[0] - aabb_min[0]) / 2
        self.half_width = (aabb_max[1] - aabb_min[1]) / 2
        self.top = aabb_max[2]

    def get_current_pos(self):
        return p.getBasePositionAndOrientation(self.id)[0]
