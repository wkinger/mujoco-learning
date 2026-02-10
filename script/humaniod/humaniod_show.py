import mujoco
import numpy as np
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/util/")
from plotter import DataCollector
import mymath.translation as translation
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/mujoco")
from mujoco_framework import ConfigBase, MuJoCoBase
import modern_robotics as mr


# Franka 示教拖动
model_directory = "/home/kuanwang/workspace/model_files/mjmodel.xml"

class ConfigHumanoid(ConfigBase):
    class Sim(ConfigBase.Sim):
        xml_scene_filename = model_directory
        dt = 0.001
        sim_time = 100
        sim_mode = "kin"  # "dyn","kin" 选择是运动学仿真还是动力学仿真 
        save_to_file = True
        # site_name = "attachment_site" # End-effector site we wish to control.
        # key_name = "home" # home位置

    class Render(ConfigBase.Render):
        is_render = True        # 是否打开渲染
        render_fps = 10        # 每step 10步，更新一次渲染
        show_left_ui = True    # 是否打开左右UI界面
        show_right_ui = True   # 是否打开左右UI界面
        

class MuJoCoHumanoid(MuJoCoBase):
    def __init__(self, cfg: ConfigHumanoid):
        super().__init__(cfg)

        # Get the dof and actuator ids for the joints we wish to control.
        # joint_names = [
        #     "joint1", "joint2", "joint3", "joint4", 
        #     "joint5", "joint6", "joint7",
        # ]
        # self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        # self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

        # Mocap body we will control with our mouse.
        # mocap_name = "target"
        # self.mocap_id = self.model.body(mocap_name).mocapid[0]

    def pre_step(self):
            pass
            
            
    def post_step(self):
        pass
        
if __name__ == "__main__":
    config = ConfigHumanoid()
    Control = MuJoCoHumanoid(config)
    Control.simulation()
    # Control.plot(save_to_file=True, filename="simulation_results.png")
