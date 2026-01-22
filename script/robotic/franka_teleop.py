import mujoco
import numpy as np
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/util/")
from plot import DataCollector, read_trajectory_data
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/mujoco")
from mujoco_framework import ConfigBase, MuJoCoBase
import numpy as np

import time
sys.path.append("/home/kuanwang/workspace/yaocao/teleop20251212/oculus_reader/robot/")
from meta_quest import MetaQuest, print_vr_data
import scipy.spatial.transform as st
epsilon = 1e-6
# Franka 示教拖动
model_directory = "/home/kuanwang/workspace/mujoco_ws/mjctrl/franka_emika_panda/scene_tau.xml"

# 使用metaquest Oculus2 进行遥操作
class ConfigFRanka(ConfigBase):
    class Sim(ConfigBase.Sim):
        xml_scene_filename = model_directory
        dt = 0.001
        sim_time = 1000
        sim_mode = "dyn"  # "dyn","kin" 选择是运动学仿真还是动力学仿真 
        save_to_file = True
        site_name = "attachment_site" # End-effector site we wish to control.
        key_name = "home" # home位置

    class Render(ConfigBase.Render):
        is_render = True        # 是否打开渲染
        render_fps = 10        # 每step 10步，更新一次渲染
        show_left_ui = True    # 是否打开左右UI界面
        show_right_ui = True   # 是否打开左右UI界面

        # Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 75.0, 50.0, 40.0, 25.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation.
Kpos: float = 0.95
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

class MuJoCoFranka(MuJoCoBase):
    def __init__(self, cfg: ConfigFRanka):
        super().__init__(cfg)
        np.set_printoptions(precision=4, suppress=True)

        # self.model.body_gravcomp[:] = float(True)
        # Get the dof and actuator ids for the joints we wish to control.
        joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7",
        ]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])
        # Compute damping and stiffness matrices.
        damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
        damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
        self.Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
        self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        self.Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)
        # Additional arrays for data collection
        self.site_quat_temp = np.zeros(4)
        self.plotter = DataCollector() 
        self.count = 0  
        self.command =  self.q0.copy()
        self.control_period = 0.05
        self.steps = int(self.control_period / self.model.opt.timestep)
        self.calc = False
        self.traj = []
        self.j = 0

         # Pre-allocate numpy arrays.
        self.jac = np.zeros((6, self.model.nv))
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.M_inv = np.zeros((self.model.nv, self.model.nv))
        self.Mx = np.zeros((6, 6))
        
        # Additional arrays for data collection
        self.site_quat_temp = np.zeros(4)
        self.mocap_quat_temp = np.zeros(4)
        self.vel_temp = np.zeros(3)  # Temporary array for velocity conversion

        self.ref_pos = self.data.site(self.site_id).xpos.copy()
        self.ref_rot = self.data.site(self.site_id).xmat.copy()
            # Mocap body we will control with our mouse.
        mocap_name = "target"
        self.mocap_id = self.model.body(mocap_name).mocapid[0]
        self.tau = 0.
        self.target_quat = self.data.mocap_quat[self.mocap_id].copy()
        self.target_pose = self.data.mocap_pos[self.mocap_id].copy()

        try:
            # 初始化VR设备
            self.quest = MetaQuest()
            print("MetaQuest VR设备初始化成功！\n")
            time.sleep(1)
        except RuntimeError as e:
            print(f"\n初始化失败：{e}")

    def pre_step(self):

        if self.count == self.steps or self.count == 0:
        # if True:
            self.quest.update()
            delta = self.quest.get_right_arm_increment()

            print(f"self.count {self.count} delta: {delta}")
            self.count = 0
            target_rot = st.Rotation.from_euler('zyx',delta[3:],degrees=True) * st.Rotation.from_matrix(self.ref_rot.reshape(3, 3))
            # print(f"target_rot: {target_rot.as_matrix()}")
            # self.target_quat = self.data.mocap_quat[self.mocap_id].copy()
            target_rot_matrix = target_rot.as_matrix().flatten()
            mujoco.mju_mat2Quat(self.target_quat, target_rot_matrix)

            self.target_pose = self.ref_pos + delta[:3]
            dx = np.array(delta[:3])

            # Spatial velocity (aka twist).
        dx = self.target_pose - self.data.site(self.site_id).xpos
        self.twist[:3] = Kpos * dx / integration_dt

        # Convert rotation matrix to quaternion for site
        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, self.target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= Kori / integration_dt
        print(f"twist: {self.twist}\n target_quat: {self.target_quat}\n site_quat: {self.site_quat}")
        # Compute end-effector Jacobian.
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # Compute the task-space inertia matrix.
        mujoco.mj_solveM(self.model, self.data, self.M_inv, np.eye(self.model.nv))
        Mx_inv = self.jac @ self.M_inv @ self.jac.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        # Compute generalized forces.
        self.tau = self.jac.T @ Mx @ (self.Kp * self.twist - self.Kd * (self.jac @ self.data.qvel[self.dof_ids]))

        # Add joint task in nullspace.
        Jbar = self.M_inv @ self.jac.T @ Mx
        ddq = Kp_null * (self.q0 - self.data.qpos[self.dof_ids]) - self.Kd_null * self.data.qvel[self.dof_ids]
        self.tau += (np.eye(self.model.nv) - self.jac.T @ Jbar.T) @ ddq

        # Add gravity compensation.
        if gravity_compensation:
            self.tau += self.data.qfrc_bias[self.dof_ids]

        # Set the control signal and step the simulation.
        # 转矩裁剪，防止失控
        np.clip(self.tau, *self.model.actuator_ctrlrange.T, out=self.tau)
        self.data.ctrl[self.actuator_ids] = self.tau[self.actuator_ids]
        self.count = self.count + 1

            
    def post_step(self):
        actual_pose = np.zeros(6)
        actual_pose[0:3] = self.data.site(self.site_id).xpos  # Position
        
        # Convert rotation matrix to quaternion, then to Euler angles
        mujoco.mju_mat2Quat(self.site_quat_temp, self.data.site(self.site_id).xmat)
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        actual_pose[3] = np.arctan2(2*(self.site_quat_temp[0]*self.site_quat_temp[1] + self.site_quat_temp[2]*self.site_quat_temp[3]), 
                                    1-2*(self.site_quat_temp[1]**2 + self.site_quat_temp[2]**2))
        actual_pose[4] = np.arcsin(2*(self.site_quat_temp[0]*self.site_quat_temp[2] - self.site_quat_temp[3]*self.site_quat_temp[1]))
        actual_pose[5] = np.arctan2(2*(self.site_quat_temp[0]*self.site_quat_temp[3] + self.site_quat_temp[1]*self.site_quat_temp[2]), 
                                    1-2*(self.site_quat_temp[2]**2 + self.site_quat_temp[3]**2))
        
        # Get target pose: position + orientation
        target_pose = np.zeros(6)
        target_pose[0:3] = self.target_pose  # Position
        
        # Mocap quaternion is already available
        self.mocap_quat_temp[:] = self.target_quat
        # Convert mocap quaternion to Euler angles
        target_pose[3] = np.arctan2(2*(self.mocap_quat_temp[0]*self.mocap_quat_temp[1] + self.mocap_quat_temp[2]*self.mocap_quat_temp[3]), 
                                    1-2*(self.mocap_quat_temp[1]**2 + self.mocap_quat_temp[2]**2))
        target_pose[4] = np.arcsin(2*(self.mocap_quat_temp[0]*self.mocap_quat_temp[2] - self.mocap_quat_temp[3]*self.mocap_quat_temp[1]))
        target_pose[5] = np.arctan2(2*(self.mocap_quat_temp[0]*self.mocap_quat_temp[3] + self.mocap_quat_temp[1]*self.mocap_quat_temp[2]),  
                                    1-2*(self.mocap_quat_temp[2]**2 + self.mocap_quat_temp[3]**2))
        self.plotter.add_data(self.data.time, actual_pose, target_pose)

if __name__ == "__main__":

    config = ConfigFRanka()
    Control = MuJoCoFranka(config)
    Control.simulation()
    Control.plot(save_to_file=True, filename="simulation_results.png")
