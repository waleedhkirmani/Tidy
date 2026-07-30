import pybullet as p


class SmallCube:
    def __init__(self, base_pos, rgba):
        self.base_pos = base_pos
        self.color = rgba
        self.id = p.loadURDF("cube_small.urdf", basePosition=base_pos)
        p.changeVisualShape(self.id, linkIndex=-1, rgbaColor=rgba)

    def get_current_pos(self):
        return p.getBasePositionAndOrientation(self.id)[0]
