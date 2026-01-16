import mujoco
import numpy as np
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/")
from util.plot import DataCollector
sys.path.append("/home/kuanwang/workspace/mujoco_ws/script/mujoco")
from mujoco_framework import ConfigBase, MuJoCoBase
import numpy as np
# Franka 示教拖动
model_directory = "/home/kuanwang/workspace/mujoco_ws/mjctrl/franka_emika_panda/scene.xml"
from scipy.optimize import minimize
from numpy.linalg import norm, solve
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import pinocchio
import time

class ConfigFRanka(ConfigBase):
    class Sim(ConfigBase.Sim):
        xml_scene_filename = model_directory
        dt = 0.001
        sim_time = 100
        sim_mode = "dyn"  # "dyn","kin" 选择是运动学仿真还是动力学仿真 
        save_to_file = True

    class Render(ConfigBase.Render):
        is_render = True        # 是否打开渲染
        render_fps = 10        # 每step 10步，更新一次渲染
        show_left_ui = True    # 是否打开左右UI界面
        show_right_ui = True   # 是否打开左右UI界面
        

class MuJoCoFranka(MuJoCoBase):    
    def __init__(self, cfg: ConfigFRanka):
        super().__init__(cfg)
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
        
        # Additional arrays for data collection
        self.site_quat_temp = np.zeros(4)
        self.mocap_quat_temp = np.zeros(4)
        self.vel_temp = np.zeros(3)  # Temporary array for velocity conversion
        self.plotter = DataCollector()   
        self.robot = pinocchio.buildModelFromUrdf("/home/kuanwang/workspace/mujoco_ws/mujoco-learning/model/franka_panda_urdf/robots/panda_arm.urdf")
        print('robot name: ' + self.robot.name)
        # <key qpos='-1.09146e-23 0.00126288 -3.32926e-07 -0.0696243 -2.28695e-05 0.192135 0.00080101 -5.53841e-09 2.91266e-07'/>
        # <key qpos='0.00296359 0.0163993 0.00368401 -0.0788281 0.259307 0.192303 -0.00312336 -4.3278e-08 1.45579e-07'/>
        # <key qpos='0.00498913 0.246686 0.00381545 -0.0800148 0.415234 0.193705 -0.00425587 3.30553e-07 -3.30357e-07'/>
        # <key qpos='0.00602759 0.424817 0.00377697 -0.0799391 0.371875 0.193076 -0.00368281 1.0024e-06 -1.11103e-07'/>
        # <key qpos='0.00773196 0.822049 0.00373852 -0.0797594 0.315405 0.192995 -0.00319569 1.50177e-06 -9.53418e-07'/>
        # <key qpos='0.00840017 1.08506 0.00374512 -0.0796342 0.304862 0.193412 -0.00331262 2.46441e-06 2.2996e-08'/>
        way_pts = [
            [-1.09146e-23, 0.00126288, -3.32926e-07, -0.0696243, -2.28695e-05, 0.192135, 0.00080101, -5.53841e-09, 2.91266e-07],
            [0.00296359, 0.0163993, 0.00368401, -0.0788281, 0.259307, 0.192303, -0.00312336, -4.3278e-08, 1.45579e-07],
            [0.00498913, 0.246686, 0.00381545, -0.0800148, 0.415234, 0.193705, -0.00425587, 3.30553e-07, -3.30357e-07],
            [0.00602759, 0.424817, 0.00377697, -0.0799391, 0.371875, 0.193076, -0.00368281, 1.0024e-06, -1.11103e-07],
            [0.00773196, 0.822049, 0.00373852, -0.0797594, 0.315405, 0.192995, -0.00319569, 1.50177e-06, -9.53418e-07],
            [0.00840017, 1.08506, 0.00374512, -0.0796342, 0.304862, 0.193412, -0.00331262, 2.46441e-06, 2.2996e-08]
        ]
           
        path_scalars = np.linspace(0, 1, len(way_pts))
        path = ta.SplineInterpolator(path_scalars, way_pts)
        vlim = np.vstack([-self.robot.velocityLimit, self.robot.velocityLimit]).T
        al = np.array([2,] * self.robot.nv)
        alim = np.vstack([-al, al]).T
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
            alim, discretization_scheme=constraint.DiscretizationType.Interpolation)
        
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc],path,solver_wrapper="seidel")
        jnt_traj = instance.compute_trajectory(0, 0)
        ts_sample = np.linspace(0, jnt_traj.get_duration(), 1000)
        self.qs_sample = jnt_traj.eval(ts_sample)
        self.index = 0
    
    def pre_step(self):
        if self.index < len(self.qs_sample):
            self.data.qpos[:7] = self.qs_sample[self.index][:7]
            self.index += 1
        else:
            self.data.qpos[:7] = self.qs_sample[-1][:7]
        time.sleep(0.01)
            
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
    Control.simulation(True, "simulation_results.png")


