import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import time
import os
import matplotlib
# 使用TkAgg后端，避免Qt依赖
matplotlib.use('TkAgg')  # 使用TkAgg后端，基于Tkinter
import matplotlib.pyplot as plt
import signal
import math
import sys

class DataCollector:
    def __init__(self):
        # 数据收集容器
        self.timestamps = []
        self.actual_x = []
        self.desired_x = []
        self.actual_z = []
        self.desired_z = []
        
    def add_data(self, timestamp, actual_x, desired_x, actual_z, desired_z):
        """收集数据点"""
        self.timestamps.append(timestamp)
        self.actual_x.append(actual_x)
        self.desired_x.append(desired_x)
        self.actual_z.append(actual_z)
        self.desired_z.append(desired_z)
    
    def plot_after_simulation(self):
        """仿真结束后显示绘图界面"""
        if not self.timestamps:
            print("没有数据可绘制")
            return
            
        print("仿真完成，开始绘制图表...")
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('Double Pendulum Control - Simulation Results')
        
        # 绘制X位置跟踪
        ax1.plot(self.timestamps, self.actual_x, 'y-', label='Actual X', linewidth=2)
        ax1.plot(self.timestamps, self.desired_x, 'r--', label='Desired X', linewidth=2)
        ax1.set_title('Position-X Tracking')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制Z位置跟踪
        ax2.plot(self.timestamps, self.actual_z, 'y-', label='Actual Z', linewidth=2)
        ax2.plot(self.timestamps, self.desired_z, 'r--', label='Desired Z', linewidth=2)
        ax2.set_title('Position-Z Tracking')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position')
        ax2.legend()
        ax2.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图表界面
        print("正在显示绘图界面...")
        plt.show()
        print("图表显示完成")

class Double_Pendulum_Control:
    def __init__(self, filename, is_show, controller_type):
        # 1. model and data
        self.model = mj.MjModel.from_xml_path(filename)
        self.data = mj.MjData(self.model)
        self.is_show = is_show
        self.running = True  # 运行状态标志
        
        # 2. 使用数据收集器替代实时绘图
        self.data_collector = DataCollector()
        self.controller_type = controller_type
        # 3. 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        
        if self.is_show:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.keyboard_cb)
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.viewer.cam.lookat = [0.0, 0.0, 0.0]
            self.viewer.cam.distance = 8.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
        
        # 4. 初始化控制器
        self.init_controller()
        print("控制器初始化完成")

    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n收到中断信号，正在退出程序...")
        self.running = False
        # 仿真中断时也绘制已有数据
        if len(self.data_collector.timestamps) > 0:
            self.data_collector.plot_after_simulation()

    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0] = 0.8
        self.data.qpos[1] = -1.6
        mj.mj_forward(self.model, self.data)
        # sensordata1-6分别代表xyz位置和xyz速度
        if self.controller_type == "cartesian":
            self.control_param = [self.data.sensordata[0], self.data.sensordata[2]]
            self.controller_type = "cartesian"
            print("使用笛卡尔空间控制")
        if self.controller_type == "joint":
            self.control_param = [self.data.sensordata[6], self.data.sensordata[8]]
            self.controller_type = "joint"
            print("使用关节空间控制")
        # 2. set the controller
        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        # 动态跟随轨迹
        freq = 0.3
        ampitude = 0.5
        delta  = ampitude * math.sin(self.data.time * 2 * math.pi * freq)
        if 10 < self.data.time < 20:
            delta = 0
        desire_command = np.array(self.control_param) + np.array([delta, delta]) 
        # print(f"当前时间：{self.data.time:.2f}, 变化量：{delta:.2f} 描述命令：{desire_command}")
        # 加入外力干扰
        if 3 < self.data.time < 5:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
            data.xfrc_applied[body_id][0] = -100
            data.xfrc_applied[body_id][2] = -100
        if self.controller_type == "cartesian":
            tau = self.cart_imp_control(desire_command)
        if self.controller_type == "joint":
            tau = self.joint_imp_control(desire_command)
        data.ctrl[0] = tau[0]
        data.ctrl[1] = tau[1]
                
        # 收集数据（不进行实时绘图）
        if self.controller_type == "cartesian":
            self.data_collector.add_data(
                self.data.time, 
                self.data.sensordata[0], 
                desire_command[0], 
                self.data.sensordata[2], 
                desire_command[1]
            )
        if self.controller_type == "joint":
            self.data_collector.add_data(
                self.data.time, 
                self.data.sensordata[6], 
                desire_command[0], 
                self.data.sensordata[8], 
                desire_command[1]
            )

    def cart_imp_control(self, cart_pos):
        """
            Md = J^(-T) * M * J^(-1)
            tau = J^T*( Kd*(Xd-X)-Bd*dX - Md*dJ*dq) + C + G
        """
        # 1. Jacobian and dot Jacobian
        q1, q2 = self.data.qpos[0], self.data.qpos[1]
        dq1, dq2 = self.data.qvel[0], self.data.qvel[1]
        # q1_sensor, q2_sensor = self.data.sensordata[6], self.data.sensordata[8]
        # v1_sensor, v2_sensor = 
        # print(f"q1: {q1}, v2: {dq2}, q1_sensor: {q1_sensor}, v2_sensor: {v2_sensor}")
        sinq1, cosq1 = math.sin(q1), math.cos(q1)
        sinq12, cosq12 = math.sin(q1 + q2), math.cos(q1 + q2)
        l1, l2 = 1, 1
        J = np.array([[-l1 * sinq1 - l2 * sinq12, -l2 * sinq12],
                    [l1 * cosq1 + l2 * cosq12, l2 * cosq12]])
        dJ = np.array([[-l1 * cosq1 * dq1 - l2 * cosq12 * (dq1 + dq2), -l2 * cosq12 * (dq1 + dq2)],
                    [-l1 * sinq1 * dq1 - l2 * sinq12 * (dq1 + dq2), -l2 * sinq12 * (dq1 + dq2)]])
        # 2. J^(-1)， J^(T), J^(-T)
        Jinv = np.linalg.pinv(J)
        JT = J.transpose()
        JinvT = Jinv.transpose()
        # 3. M, C + G
        M = np.zeros((2, 2))
        mj.mj_fullM(self.model, M, self.data.qM)
        Fbias = self.data.qfrc_bias
        # 4. Md, Bd, Kd
        Md = JinvT @ M @ Jinv
        Bd = 100 * np.eye(2)
        Kd = 820 * np.eye(2)
        # 5. controller
        Xd = np.array([cart_pos[0], cart_pos[1]])
        X  = np.array([self.data.sensordata[0], self.data.sensordata[2]])
        dX = np.array([self.data.sensordata[3], self.data.sensordata[5]])
        dq = np.array([dq1, dq2])
        tau = JT @ (Kd @ (Xd - X) - Bd @ dX - Md @ dJ @ dq) + Fbias
        return tau
    def joint_imp_control(self, joint_pos):
        """
            Md =  M 
            tau = Kd*(qd-q) - Bd*dq + C + G
        """
        # 3. M, C + G
        M = np.zeros((2, 2))
        mj.mj_fullM(self.model, M, self.data.qM)
        Fbias = self.data.qfrc_bias
        # 4. Md, Bd, Kd
        Md =  M 
        Bd = 100 * np.eye(2)
        Kd = 420 * np.eye(2)
        # 5. controller
        qd = np.array([joint_pos[0], joint_pos[1]])
        q  = np.array([self.data.sensordata[6], self.data.sensordata[8]])
        dq = np.array([self.data.sensordata[7], self.data.sensordata[9]])
        tau = Kd @ (qd - q) - Bd @ dq + Fbias
        return tau

    def main(self):
        sim_start = time.time()
        sim_end = 5.0  # 缩短仿真时间便于观察
        
        try:
            while self.running and (time.time() - sim_start < sim_end):
                step_start = time.time()
                # loop_num 是每次循环的步数，这里设置为50，即0.002*50 = 0.1s，意思是仿真是可视化的50倍
                loop_num, loop_count = 50, 0
                while self.running and loop_count < loop_num:
                    loop_count = loop_count + 1
                    mj.mj_step(self.model, self.data)
                    # 每步进后检查一次运行状态
                    if not self.running:
                        break
                
                if not self.running:
                    break
                    
                # 2. GUI show
                if self.is_show:
                    if self.viewer.is_running():
                        self.viewer.sync()
                    else:
                        break
                # 3. sleep for next period (使用短时间睡眠并轮询)
                step_next_delta = self.model.opt.timestep * loop_num - (time.time() - step_start)
                if step_next_delta > 0:
                    time.sleep(min(step_next_delta, 0.01))
                    
        except KeyboardInterrupt:
            # 直接捕获KeyboardInterrupt异常
            print("\n捕获到KeyboardInterrupt异常，正在退出程序...")
            self.running = False
        
        # 4. 确保Viewer正确关闭
        if self.is_show and hasattr(self, 'viewer'):
            self.viewer.close()
        
        # 5. 仿真结束后显示绘图界面
        self.data_collector.plot_after_simulation()

    def keyboard_cb(self, keycode):
        if chr(keycode) == ' ':
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.init_controller()
            # 清空之前收集的数据
            self.data_collector = DataCollector()

if __name__ == "__main__":
    xml_path = "/home/kuanwang/workspace/mujoco_ws/env/double_pendulum_force.xml"
    is_show = True
    Control = Double_Pendulum_Control(xml_path, is_show, controller_type="cartesian")
    Control.main()