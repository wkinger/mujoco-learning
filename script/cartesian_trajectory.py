import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import time
import os
import matplotlib.pyplot as plt  # 添加绘图库
import math

class Double_Pendulum_Control:
    def __init__(self, filename, is_show):
        # 1. model and data
        self.model = mj.MjModel.from_xml_path(filename)
        self.data = mj.MjData(self.model)
        self.is_show = is_show
        if self.is_show:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.keyboard_cb)
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.viewer.cam.lookat = [0.0, 0.0, 0.0]
            self.viewer.cam.distance = 8.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
        # 2. init Controller
        self.init_controller()
        # 3. init data recording
        self.init_data_recording()
        print("init controller")

    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0] = -0.5
        self.data.qpos[1] = 1.0
        
        # 2. 计算初始末端执行器位置（使用FK而不是传感器）
        l1, l2 = 1, 1
        q1, q2 = self.data.qpos[0], self.data.qpos[1]
        ee_x = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
        ee_z = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
        
        # 3. 设置圆形轨迹参数
        radius = 0.4  # 减小半径以确保轨迹在工作空间内
        self.circle_param = [ee_x - radius, ee_z, radius]
        
        # 4. 初始化运动学相关参数
        self.l1, self.l2 = 1, 1
        self.max_iterations = 100  # 最大迭代次数
        self.delta_pos_threshold = 1e-4  # 位置误差阈值
        self.alpha = 0.3  # 学习率，用于IK迭代
        self.jacobian_damping = 0.01  # 雅可比矩阵阻尼因子
        
        # 5. set the controller
        mj.set_mjcb_control(self.controller)

    def init_data_recording(self):
        # 初始化数据记录
        self.recorded_time = []
        self.recorded_q1 = []
        self.recorded_q2 = []
        self.recorded_ee_x = []
        self.recorded_ee_z = []
        self.recorded_ee_x_des = []
        self.recorded_ee_z_des = []

    def get_ee_position(self):
        """获取末端执行器位置（使用正运动学计算）"""
        q1, q2 = self.data.qpos[0], self.data.qpos[1]
        x = self.l1 * math.cos(q1) + self.l2 * math.cos(q1 + q2)
        z = self.l1 * math.sin(q1) + self.l2 * math.sin(q1 + q2)
        return np.array([x, z])

    def robot_ik(self, cart_pos, current_q):
        """
        改进的迭代逆运动学求解器
        :param cart_pos: 目标笛卡尔位置 [x, z]
        :param current_q: 当前关节位置 [q1, q2]
        :return: 求解的关节位置 [q1, q2]
        """
        q1, q2 = current_q[0], current_q[1]
        
        # 迭代求解IK
        for i in range(self.max_iterations):
            # 计算当前末端执行器位置
            current_ee = self.get_ee_position()
            
            # 计算位置误差
            delta_pos = np.array([cart_pos[0] - current_ee[0],
                                cart_pos[1] - current_ee[1]])
            
            # 检查位置误差是否足够小
            if np.linalg.norm(delta_pos) < self.delta_pos_threshold:
                break
            
            # 计算雅可比矩阵
            sinq1, cosq1 = math.sin(q1), math.cos(q1)
            sinq12, cosq12 = math.sin(q1 + q2), math.cos(q1 + q2)
            
            J = np.array([[-self.l1 * sinq1 - self.l2 * sinq12, -self.l2 * sinq12],
                        [self.l1 * cosq1 + self.l2 * cosq12,  self.l2 * cosq12]])
            
            # 使用阻尼最小二乘法求解关节速度
            J_pinv = np.linalg.inv(J.T @ J + self.jacobian_damping * np.eye(2)) @ J.T
            dq = J_pinv @ delta_pos
            
            # 更新关节角度
            q1 += self.alpha * dq[0]
            q2 += self.alpha * dq[1]
        
        # 确保关节角度在合理范围内（-π到π）
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        q2 = math.atan2(math.sin(q2), math.cos(q2))
        
        return np.array([q1, q2])

    def controller(self, model, data):
        # 1. 生成笛卡尔空间目标轨迹（使用固定角速度）
        omega = 2  # 角速度 (rad/s)
        x0, z0, r = self.circle_param[0], self.circle_param[1], self.circle_param[2]
        cart_des = [x0 + r * math.cos(omega * data.time), z0 + r * math.sin(omega * data.time)]
        
        # 2. 求解逆运动学
        current_q = np.array([data.qpos[0], data.qpos[1]])
        joint_des = self.robot_ik(cart_des, current_q)
        
        # 3. 检查关节位置是否有效
        if not np.all(np.isfinite(joint_des)):
            print("Warning: Invalid joint positions calculated by IK")
            return
        
        # 4. 使用位置控制器
        data.ctrl[0] = joint_des[0]  # joint1位置控制
        data.ctrl[3] = joint_des[1]  # joint2位置控制
        
        # 5. 设置位置控制器的KP值
        model.actuator_gainprm[0, 0] = 600.0  # pos_ctrl1的KP
        model.actuator_gainprm[3, 0] = 600.0  # pos_ctrl2的KP
        
        # 6. 添加速度阻尼
        data.ctrl[1] = -0.5 * data.qvel[0]  # joint1速度阻尼
        data.ctrl[4] = -0.5 * data.qvel[1]  # joint2速度阻尼
        
        # 7. 检查控制输出是否有效
        if not np.all(np.isfinite(data.ctrl)):
            print("Warning: Invalid control output")
            data.ctrl[:] = 0.0

    def record_data(self, data):
        """记录数据用于后续分析"""
        self.recorded_time.append(data.time)
        self.recorded_q1.append(data.qpos[0])
        self.recorded_q2.append(data.qpos[1])
        
        # 记录实际末端位置
        ee_pos = self.get_ee_position()
        self.recorded_ee_x.append(ee_pos[0])
        self.recorded_ee_z.append(ee_pos[1])
        
        # 记录期望末端位置
        omega = 0.5
        x0, z0, r = self.circle_param[0], self.circle_param[1], self.circle_param[2]
        cart_des = [x0 + r * math.cos(omega * data.time), z0 + r * math.sin(omega * data.time)]
        self.recorded_ee_x_des.append(cart_des[0])
        self.recorded_ee_z_des.append(cart_des[1])

    def plot_results(self):
        """绘制仿真结果"""
        plt.figure(figsize=(12, 8))
        
        # 1. 绘制关节角度随时间变化
        plt.subplot(2, 2, 1)
        plt.plot(self.recorded_time, self.recorded_q1, label='q1 (rad)')
        plt.plot(self.recorded_time, self.recorded_q2, label='q2 (rad)')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angles (rad)')
        plt.title('Joint Angles vs Time')
        plt.legend()
        plt.grid(True)
        
        # 2. 绘制末端执行器轨迹（笛卡尔空间）
        plt.subplot(2, 2, 2)
        plt.plot(self.recorded_ee_x_des, self.recorded_ee_z_des, 'r--', label='Desired Trajectory')
        plt.plot(self.recorded_ee_x, self.recorded_ee_z, 'b-', label='Actual Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Z Position (m)')
        plt.title('End-Effector Trajectory')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # 3. 绘制末端执行器位置随时间变化
        plt.subplot(2, 2, 3)
        plt.plot(self.recorded_time, self.recorded_ee_x_des, 'r--', label='Desired X')
        plt.plot(self.recorded_time, self.recorded_ee_x, 'b-', label='Actual X')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.title('End-Effector X Position vs Time')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.recorded_time, self.recorded_ee_z_des, 'r--', label='Desired Z')
        plt.plot(self.recorded_time, self.recorded_ee_z, 'b-', label='Actual Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Z Position (m)')
        plt.title('End-Effector Z Position vs Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('cartesian_trajectory_results.png')
        plt.show()

    def main(self):
        sim_start, sim_end = time.time(), 20.0  # 延长仿真时间以观察完整轨迹
        while time.time() - sim_start < sim_end:
            step_start = time.time()
            loop_num, loop_count = 50, 0
            # 1. running for 0.002*50 = 0.1s
            while loop_count < loop_num:
                loop_count = loop_count + 1
                mj.mj_step(self.model, self.data)
            # 2. GUI show
            if self.is_show:
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    break
            # 3. 记录数据
            self.record_data(self.data)
            # 4. sleep for next period
            step_next_delta = self.model.opt.timestep * loop_count - (time.time() - step_start)
            if step_next_delta > 0:
                time.sleep(step_next_delta)
        
        if self.is_show:
            self.viewer.close()
        
        # 绘制结果
        self.plot_results()

    def keyboard_cb(self, keycode):
        if chr(keycode) == ' ':
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.init_controller()
            self.init_data_recording()  # 重置数据记录


if __name__ == "__main__":
    xml_path = "/home/user/workspace/mujoco_ws/env/double_pendulum.xml"
    is_show = True
    Control = Double_Pendulum_Control(xml_path, is_show)
    Control.main()