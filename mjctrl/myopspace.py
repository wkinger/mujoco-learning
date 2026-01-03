import mujoco
import numpy as np
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/ulits/")
from plot import DataCollector
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/")
from mujoco_framework import ConfigBase, MuJoCoBase
import numpy as np

model_directory =  "kuka_iiwa_14/scene.xml"

class ConfigFRanka(ConfigBase):
    class Sim(ConfigBase.Sim):
        xml_scene_filename = model_directory
        dt = 0.001
        sim_time = 50
        sim_mode = "dyn"  # "dyn","kin" 选择是运动学仿真还是动力学仿真 

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
        # Compute damping and stiffness matrices.
        damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
        damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
        self.Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
        self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        self.Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

        # End-effector site we wish to control.
        site_name = "attachment_site"
        self.site_id = self.model.site(site_name).id

        # Get the dof and actuator ids for the joints we wish to control.
        joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7",
        ]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        key_name = "home"
        self.key_id = self.model.key(key_name).id
        self.q0 = self.model.key(key_name).qpos

        # Mocap body we will control with our mouse.
        mocap_name = "target"
        self.mocap_id = self.model.body(mocap_name).mocapid[0]

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
        self.plotter = DataCollector()   

    def pre_step(self):
            # Spatial velocity (aka twist).
            dx = self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos
            self.twist[:3] = Kpos * dx / integration_dt
            
            # Convert rotation matrix to quaternion for site
            mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
            mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
            mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj)
            mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
            self.twist[3:] *= Kori / integration_dt

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
            tau = self.jac.T @ Mx @ (self.Kp * self.twist - self.Kd * (self.jac @ self.data.qvel[self.dof_ids]))

            # Add joint task in nullspace.
            Jbar = self.M_inv @ self.jac.T @ Mx
            ddq = Kp_null * (self.q0 - self.data.qpos[self.dof_ids]) - self.Kd_null * self.data.qvel[self.dof_ids]
            tau += (np.eye(self.model.nv) - self.jac.T @ Jbar.T) @ ddq

            # Add gravity compensation.
            if gravity_compensation:
                tau += self.data.qfrc_bias[self.dof_ids]

            # Set the control signal and step the simulation.
            # 转矩裁剪，防止失控
            np.clip(tau, *self.model.actuator_ctrlrange.T, out=tau)
            self.data.ctrl[self.actuator_ids] = tau[self.actuator_ids]
            mujoco.mj_step(self.model, self.data)
            
    def post_step(self):
        # Data collection for plotting - FIXED VERSION
        # Get actual pose: position + orientation (as Euler angles)
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
        target_pose[0:3] = self.data.mocap_pos[self.mocap_id]  # Position
        
        # Mocap quaternion is already available
        self.mocap_quat_temp[:] = self.data.mocap_quat[self.mocap_id]
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