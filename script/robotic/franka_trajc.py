import mujoco
import numpy as np
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/util/")
from plot import DataCollector, read_trajectory_data
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/mujoco")
from mujoco_framework import ConfigBase, MuJoCoBase
import numpy as np
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/trajctory")
from wkdoubleS_trajectory_mulity import DoubleSCurveTrajectoryGenerator
import time

epsilon = 1e-6
# Franka 示教拖动
model_directory = "/home/kuanwang/workspace/mujoco_ws/mjctrl/franka_emika_panda/scene.xml"


class ConfigFRanka(ConfigBase):
    class Sim(ConfigBase.Sim):
        xml_scene_filename = model_directory
        dt = 0.001
        sim_time = 100
        sim_mode = "dyn"  # "dyn","kin" 选择是运动学仿真还是动力学仿真 
        save_to_file = True
        site_name = "attachment_site" # End-effector site we wish to control.
        key_name = "home" # home位置

    class Render(ConfigBase.Render):
        is_render = True        # 是否打开渲染
        render_fps = 10        # 每step 10步，更新一次渲染
        show_left_ui = True    # 是否打开左右UI界面
        show_right_ui = True   # 是否打开左右UI界面
        
constrains = np.array(([0,10,0,0,  5,-5,10,-10,30,-30],
                        [0,10,0,0,  5,-5,10,-10,30,-30],
                        [0,13,0,0,  5,-5,10,-10,30,-30],
                        [30,10,0,0, 5,-5,10,-10,30,-30],
                        [-10,10,0,0,5,-5,10,-10,30,-30],
                        [0,11,0,0,  5,-5,10,-10,30,-30],
                        [2,10,0,0,  5,-5,10,-10,30,-30]
                        ), dtype=float)

class MuJoCoFranka(MuJoCoBase):
    def __init__(self, cfg: ConfigFRanka):
        super().__init__(cfg)
        self.model.body_gravcomp[:] = float(True)
        # Get the dof and actuator ids for the joints we wish to control.
        joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7",
        ]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])
        
        # Additional arrays for data collection
        self.site_quat_temp = np.zeros(4)
        self.plotter = DataCollector() 
        self.count = 0  
        self.command =  self.q0.copy()
        self.control_period = 0.1
        self.steps = self.control_period / self.model.opt.timestep
        self.calc = False
        self.traj = []
        self.j = 0
        # 初始化轨迹规划器
        self.planner = DoubleSCurveTrajectoryGenerator(7, constrains[:,0], constrains[:,1], constrains[:,2], constrains[:,3], \
                    constrains[:,4], constrains[:,5], constrains[:,6], constrains[:,7], constrains[:,8], constrains[:,9])
    def pre_step(self):
        start_time = 1.
        if self.data.time >= start_time:
            command = np.array([1,-0.5,-1,-0.5,-0.4,0.9,-0.5])
            if self.calc == False:
                constrains[:,0] = self.data.qpos.copy()
                constrains[:,1] = command
                constrains[:,2] = np.zeros(self.planner.dof)
                constrains[:,3] = np.zeros(self.planner.dof)
                self.planner.reset(constrains[:,0], constrains[:,1], constrains[:,2], constrains[:,3], \
                    constrains[:,4], constrains[:,5], constrains[:,6], constrains[:,7], constrains[:,8], constrains[:,9])
                total_time = self.planner.multi_double_s_curve_trajectory(self.model.opt.timestep, start_time, start_time + 5)
                self.traj = self.planner.traj_pos
                self.calc = True
            if self.j < self.traj.shape[1] - 1:
                self.command = self.traj[:,self.j]
                print(f"self.traj: {(self.traj.shape[1])} j {self.j}  self.calc {self.calc}")
                print(f"self.command: {self.command} \n cur {self.data.qpos.copy()}")
            else:
                self.command = self.traj[:,-1]

            self.j += 1

                # delta = 0.5 * np.sin(self.data.time * np.pi * 0.1 * 2)
                # self.command = self.q0 + delta * np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

            self.data.ctrl[self.actuator_ids] = self.command[self.dof_ids]
        self.count = self.count + 1
            
    def post_step(self):
        # Data collection for plotting - FIXED VERSION
        # Get actual pose: position + orientation (as Euler angles)
        sensor = self.data.sensordata.copy()
        # print(f"sensor: {sensor}\n acutual_pose: {actual_pose}")
        self.plotter.add_data(self.data.time, sensor, self.command.copy())

if __name__ == "__main__":


    # t_list1, pos1, vel1, acc1, jerk1 = planner.get_profile(0.01,3)
    # for i in range(len(t_list)):
    #     print(f"t = {t_list[i]:.4f}, q = {pos[i]:.4f}, v = {vel[i]:.4f}, a = {acc[i]:.4f}, j = {jerk[i]:.4f}")
    # planner.plot_all_trajectories(t_list, pos, vel, acc, jerk)
   

    config = ConfigFRanka()
    Control = MuJoCoFranka(config)
    Control.simulation()
    Control.plot(save_to_file=True, filename="simulation_results.png")
