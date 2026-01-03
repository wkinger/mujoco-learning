from mujoco_framework import ConfigBase, MuJoCoBase
import numpy as np

model_directory =  "/home/kuanwang/workspace/mujoco_ws/mujoco_menagerie-main/universal_robots_ur5e/ur5e.xml"

class ConfigUR5(ConfigBase):
    class Sim(ConfigBase.Sim):
        xml_scene_filename = model_directory
        dt = 0.001
        sim_time = 50
        sim_mode = "kin"  # "dyn","kin" 选择是运动学仿真还是动力学仿真 

    class Render(ConfigBase.Render):
        is_render = True        # 是否打开渲染
        render_fps = 10         # 每step 10步，更新一次渲染
        show_left_ui = True    # 是否打开左右UI界面
        show_right_ui = True   # 是否打开左右UI界面
        # 相机参数，lookat, distance, azimuth, elevation
        # cam_para = [0.012768, -0.000000, 1.254336, 10, 90, -5]  
        


class MuJoCoUR5(MuJoCoBase):
    def __init__(self, cfg: ConfigUR5):
        super().__init__(cfg)

    def pre_step(self):
        key_name = "home"
        pos_des = self.model.key("home").qpos
        vel_des = np.zeros_like(pos_des)
        tau_ff = np.zeros_like(pos_des)
        kp = np.diag([20, 200, 200, 200, 20, 20])
        kd = np.diag([2, 20, 20, 20, 2, 2])
        self.pvt_control(pos_des, vel_des, tau_ff, kp, kd)

    def pvt_control(self, pos_des, vel_des, trq_ff, kp, kd):
        self.data.ctrl = kp @ (pos_des - self.data.qpos) + kd @ (vel_des - self.data.qvel) + trq_ff

if __name__ == "__main__":
    config = ConfigUR5()
    Control = MuJoCoUR5(config)
    Control.simulation()
