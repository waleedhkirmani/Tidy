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
