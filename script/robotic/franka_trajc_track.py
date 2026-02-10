import numpy as np
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/util/")
from plotter import DataCollector, read_trajectory_data
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/mujoco")
from mujoco_framework import ConfigBase, MuJoCoBase
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/trajctory")
from TD_planner import MultiAxisConstrainedPositionPlanner, MultiAxisFirstOrderTD

epsilon = 1e-6
# Franka 轨迹跟踪
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
        self.control_period = 0.05
        self.steps = int(self.control_period / self.model.opt.timestep)
        self.calc = False
        self.j = 0

        file_path = "trajData_wk-1-1-5.txt"
        positions, velocities = read_trajectory_data(file_path)
        if len(positions) == 0:
            print("错误：未读取到有效数据")
            exit(1)
        print(f"成功读取 {len(positions)} 个数据点")
        self.joint_pos_raw = positions.T
        self.offset = np.array([-0.5, 0.8, 0.5, -2.6, 0.0, 1.4, 0.0]) #角度偏置

        # 创建单轴四阶LTD位置规划器（保持向后兼容）
        self.planner = MultiAxisConstrainedPositionPlanner(
            num_axes=7, 
            omega_c=150, 
            sample_time=0.001, 
            initial_positions = self.joint_pos_raw[:,0] + np.array([-0.5, 0.8, 0.5, -2.6, 0.0, 1.4, 0.0]), 
            max_velocity = 1, 
            max_acceleration = 5, 
            max_jerk = 15
        )
        # self.planner =  MultiAxisFirstOrderTD(num_axes=7, initial_positions = self.joint_pos_raw[:,0] + self.offset, k=100, dt=0.001)


    def pre_step(self):
        start_time = 1.
        command = self.q0.copy() + self.offset
        if self.data.time < start_time:
            command = self.joint_pos_raw[:,0] + self.offset
            self.command = command
        if self.count % self.steps == 0 and self.data.time >= start_time :
            if self.data.time > 2:
                if self.j < self.joint_pos_raw.shape[1] - 1:
                    # delta = 1 * np.sin(self.j * np.pi * 0.01 * 2)
                    # command = self.q0 + delta * np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])+ np.random.normal(0, 0.3, 7)
                    command = self.joint_pos_raw[:,self.j] + self.offset
                    # print(f"self.joint_pos_raw: {(self.joint_pos_raw.shape[1])} j {self.j}  self.calc {self.calc}")
                    # print(f"self.command: {self.command} \n cur {self.data.qpos.copy()}")
                else:
                    command = self.joint_pos_raw[:,-1] + self.offset
                self.j += 1

            else:
                command = self.joint_pos_raw[:,0] + self.offset
            self.command = command
        # print(f"self.command: {self.command}")

        traj, vel,_,_ = self.planner.update(self.command)
        # traj, vel = self.planner.update(self.command)
        # print(f"vel: {vel}")
        self.data.ctrl[self.actuator_ids] = traj[self.dof_ids]
        self.count = self.count + 1
            
    def post_step(self):
        # Data collection for plotting - FIXED VERSION
        # Get actual pose: position + orientation (as Euler angles)
        sensor = self.data.sensordata.copy()
        # print(f"sensor: {sensor}\n acutual_pose: {self.command.copy()}")
        self.plotter.add_data(self.data.time, sensor[:7], self.command.copy())

if __name__ == "__main__":


    # t_list1, pos1, vel1, acc1, jerk1 = planner.get_profile(0.01,3)
    # for i in range(len(t_list)):
    #     print(f"t = {t_list[i]:.4f}, q = {pos[i]:.4f}, v = {vel[i]:.4f}, a = {acc[i]:.4f}, j = {jerk[i]:.4f}")
    # planner.plot_all_trajectories(t_list, pos, vel, acc, jerk)
   

    config = ConfigFRanka()
    Control = MuJoCoFranka(config)
    Control.simulation()
    Control.plot(save_to_file=True, filename="simulation_results.png")
