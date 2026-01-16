import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import time
import os
import matplotlib.pyplot as plt  # 添加绘图库


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
        self.traj = np.array([
            [0.0,  0,    0.0],      # t=0s: 两个关节都在0弧度位置
            [2.0,  np.pi/2,  0.0],    # t=2s: 关节1到π/2，关节2保持0
            [4.0,  np.pi/2,  np.pi],  # t=5s: 关节2到π，关节1保持π/2
            [6.0,  -np.pi/4, np.pi/2], # t=8s: 关节1到-π/4，关节2到π/2
            [7.0,  -np.pi/5, np.pi/3], # t=8s: 关节1到-π/4，关节2到π/2
            [8.0,  -np.pi/6, np.pi/6], # t=8s: 关节1到-π/4，关节2到π/2
            [9.0,  0, np.pi/2], # t=8s: 关节1到-π/4，关节2到π/2
            [10.0,  np.pi/4, -np.pi/2], # t=8s: 关节1到-π/4，关节2到π/2
            [11.0,  np.pi/2, -np.pi], # t=8s: 关节1到-π/4，关节2到π/2
        ])
        self.init_controller()
        # 3. 初始化数据记录
        self.init_data_recording()
        print("init controller")

    def init_data_recording(self):
        """初始化数据记录变量"""
        self.time_history = []          # 时间序列
        self.q1_actual = []             # 关节1实际位置
        self.q2_actual = []             # 关节2实际位置
        self.q1_ref = []                # 关节1目标位置
        self.q2_ref = []                # 关节2目标位置
        self.error_q1 = []              # 关节1跟踪误差
        self.error_q2 = []              # 关节2跟踪误差

    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0], self.data.qpos[1] = self.traj[0, 1], self.traj[0, 2]
        # 2. set the controller param
        self.set_controller_param("trq_mode", 0.0, 0.0)
        # 3. set the controller
        mj.set_mjcb_control(self.controller)
        
# XML中配置，pos_mode为控制器1，vel_mode为控制器2，trq_mode为控制器3
# actuator_gainprm[X, Y]这个二维数组，X代表第几个控制器，Y代表参数
# actuator_biasprm[X, Y]这个二维数组，X代表第几个控制器，Y代表参数
    def set_controller_param(self, ctrl_mode, kp, kv):
        if ctrl_mode == "trq_mode":
            self.ctrl_mode = ctrl_mode
            self.model.actuator_gainprm[0, 0] = 0
            self.model.actuator_biasprm[0, 1] = 0
            self.model.actuator_gainprm[1, 0] = 0
            self.model.actuator_biasprm[1, 2] = 0

    ### controller designer
    def controller(self, model, data):
        # 1. trajectory planning
        q1, q2 = np.zeros((2,)), np.zeros((2,))
        for i in range(self.traj.shape[0]-1):
            if self.traj[i, 0] <= self.data.time < self.traj[i+1, 0]:
                q1 = self.generate_trajectory(self.traj[i, 0], self.traj[i + 1, 0],
                                            self.traj[i, 1], self.traj[i + 1, 1], data.time)
                q2 = self.generate_trajectory(self.traj[i, 0], self.traj[i + 1, 0],
                                            self.traj[i, 2], self.traj[i + 1, 2], data.time)
                break
        if self.data.time >= self.traj[-1, 0]:
            q1 = np.array([self.traj[-1, 1], 0.0])
            q2 = np.array([self.traj[-1, 2], 0.0])

        # 2. tracking controller
        Kp, Kv = 500, 50
        q1_ref, q2_ref = q1[0], q2[0]
        dq1_ref, dq2_ref = q1[1], q2[1]
        data.qfrc_applied[0] = Kp * (q1_ref - data.qpos[0]) + Kv * (dq1_ref - data.qvel[0])
        data.qfrc_applied[1] = Kp * (q2_ref - data.qpos[1]) + Kv * (dq2_ref - data.qvel[1])
        # 3. 记录当前时刻数据
        self.record_data(q1_ref, q2_ref, data.time, data.qpos, data.qvel)

    def record_data(self, q1_ref, q2_ref, time, qpos, qvel):
        """记录当前时刻的跟踪数据"""
        self.time_history.append(time)
        self.q1_actual.append(qpos[0])
        self.q2_actual.append(qpos[1])
        self.q1_ref.append(q1_ref)
        self.q2_ref.append(q2_ref)
        self.error_q1.append(q1_ref - qpos[0])
        self.error_q2.append(q2_ref - qpos[1])

    ### trajectory generation
    def generate_trajectory(self, t0, t1, x0, x1, t):
        """
            cubic trajectory: x(t) = a0 + a1*t + a2*t^2 + a3*t^3
            a0 = ( x0*t1^2*(t1-3*t0) + x1*t0^2*(3*t1-t0) )/ (t1-t0)^3
            a1 = 6*t0*t1*(x0-x1) / (t1-t0)^3
            a2 = 3*(t0+t1)*(x1-x0) / (t1-t0)^3
            a3 = 2*(x0-x1) / (t1-t0)^3
        """
        rate = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        a0 = x0
        a1 = 0
        a2 = 3 * (x1 - x0)
        a3 = 2 * (x0 - x1)
        x = a0 + a1 * rate + a2 * (rate ** 2) + a3 * (rate ** 3)
        dx = a1 + 2 * a2 * rate + 3 * a3 * (rate ** 2)
        return np.array([x, dx])
    
    def plot_trajectory_results(self):
        """绘制轨迹跟踪曲线和误差曲线"""
        # 转换为numpy数组
        time_array = np.array(self.time_history)
        q1_actual_array = np.array(self.q1_actual)
        q2_actual_array = np.array(self.q2_actual)
        q1_ref_array = np.array(self.q1_ref)
        q2_ref_array = np.array(self.q2_ref)
        error_q1_array = np.array(self.error_q1)
        error_q2_array = np.array(self.error_q2)
        
        # 创建绘图窗口，2个子图（2行1列）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. 两个关节的轨迹跟踪绘制在同一幅图中
        ax1.plot(time_array, q1_ref_array, 'r--', linewidth=2, label='Joint 1 Target')
        ax1.plot(time_array, q1_actual_array, 'r-', linewidth=1.5, label='Joint 1 Actual')
        ax1.plot(time_array, q2_ref_array, 'b--', linewidth=2, label='Joint 2 Target')
        ax1.plot(time_array, q2_actual_array, 'b-', linewidth=1.5, label='Joint 2 Actual')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (rad)')
        ax1.set_title('Joint Trajectory Tracking')
        ax1.grid(True)
        ax1.legend()
        
        # 2. 两个关节的跟踪误差绘制在同一幅图中
        ax2.plot(time_array, error_q1_array, 'r-', linewidth=1.5, label='Joint 1 Error')
        ax2.plot(time_array, error_q2_array, 'b-', linewidth=1.5, label='Joint 2 Error')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (rad)')
        ax2.set_title('Joint Tracking Error')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('joint_trajectory_tracking_results.png')  # 保存图片
        plt.show()  # 显示图片

    def main(self):
        sim_start, sim_end = time.time(), 15.0  # 延长仿真时间以覆盖完整轨迹
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
                    # self.viewer.cam.lookat[0] = self.data.qpos[0]
                    self.viewer.sync()
                else:
                    break
            # 3. sleep for next period
            step_next_delta = self.model.opt.timestep * loop_count - (time.time() - step_start)
            if step_next_delta > 0:
                time.sleep(step_next_delta)
        
        # 4. 仿真结束后绘制轨迹跟踪结果
        self.plot_trajectory_results()
        
        if self.is_show:
            self.viewer.close()

    def keyboard_cb(self, keycode):
        if chr(keycode) == ' ':
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.init_controller()
            # 重置数据记录
            self.init_data_recording()


if __name__ == "__main__":
    xml_path = "/home/user/workspace/mujoco_ws/env/double_pendulum.xml"
    is_show = True
    Control = Double_Pendulum_Control(xml_path, is_show)
    Control.main()