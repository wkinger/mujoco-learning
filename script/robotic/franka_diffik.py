import mujoco
import numpy as np
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/util/")
from plot import DataCollector
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/mujoco")
from mujoco_framework import ConfigBase, MuJoCoBase
# Franka 迭代逆解+位置控制
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
        
# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785
class MuJoCoFranka(MuJoCoBase):
    def __init__(self, cfg: ConfigFRanka):
        super().__init__(cfg)
        self.model.body_gravcomp[:] = float(gravity_compensation)
        # Get the dof and actuator ids for the joints we wish to control.
        joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7",
        ]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

        # Mocap body we will control with our mouse.
        mocap_name = "target"
        self.mocap_id = self.model.body(mocap_name).mocapid[0]

        # Pre-allocate numpy arrays.
        self.jac = np.zeros((6, self.model.nv))
        self.diag = damping * np.eye(6)
        self.eye = np.eye(self.model.nv)
        self.twist = np.zeros(6)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        
        # Additional arrays for data collection
        self.site_quat_temp = np.zeros(4)
        self.mocap_quat_temp = np.zeros(4)
        self.vel_temp = np.zeros(3)  # Temporary array for velocity conversion
        self.plotter = DataCollector() 
        self.count = 0  

    def pre_step(self):
        # Spatial velocity (aka twist).
        dx = self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos
        self.twist[:3] = Kpos * dx / integration_dt
        mujoco.mju_mat2Quat(self.site_quat_temp, self.data.site(self.site_id).xmat)
        # 计算四元数的共轭（逆旋转）
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat_temp)
        mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj)
        # 将四元数误差转换为角速度形式的姿态误差
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= Kori / integration_dt

        # Jacobian.
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # Damped least squares.
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.twist)

        # Nullspace control biasing joint velocities towards the home configuration.
        dq += (self.eye - np.linalg.pinv(self.jac) @ self.jac) @ (Kn * (self.q0 - self.data.qpos[self.dof_ids]))

        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = self.data.qpos.copy()  # Note the copy here is important.
        mujoco.mj_integratePos(self.model, q, dq, integration_dt)
        np.clip(q, *self.model.jnt_range.T, out=q)
        if self.count % 500 == 0:
            print(f"twist: {self.twist}\n dq {dq}\nq {q}\n targetpos {self.data.mocap_pos[self.mocap_id]}")
        # Set the control signal and step the simulation.
        if self.count < 1000:
            q = np.array([-0.05615562 ,-0.18599034 , 0.0523708,  -2.10141468 , 0.01024992,  1.91443829, -0.77720628])
        self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
        self.count += 1 
            
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
    Control.plot(True, "simulation_results.png")