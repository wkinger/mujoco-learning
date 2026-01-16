import numpy as np
import math

import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import time
import os
import matplotlib.pyplot as plt  # 添加绘图库
import signal  # 添加信号处理库


class Double_Pendulum_Control:
    def __init__(self, filename, is_show):
        # 1. model and data
        self.model = mj.MjModel.from_xml_path(filename)
        self.data = mj.MjData(self.model)
        self.is_show = is_show
        self.running = True  # 运行状态标志
        
        # 2. 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        
        if self.is_show:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.keyboard_cb)
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.viewer.cam.lookat = [0.0, 0.0, 0.0]
            self.viewer.cam.distance = 8.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
        # 3. init Controller
        self.init_controller()
        # 4. 初始化数据记录
        self.init_data_recording()
        print("init controller")

    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n收到中断信号，正在退出程序...")
        self.running = False
        
    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0] = 0
        self.data.qpos[1] = 0
        # 2. set the controller param
        self.set_controller_param("trq_mode", 0.0, 0.0)
        # 3. set the controller
        mj.set_mjcb_control(self.controller)
        # 4. 初始化积分项
        self.integral_error = np.zeros(2)  # 积分误差累积
        self.Kp = 500 * np.eye(2)
        self.Ki = 100 * np.eye(2)  # 积分增益矩阵
        self.Kd = 10 * np.eye(2)
        self.integral_limit = np.array([1.0, 1.0])  # 积分饱和限幅
        # 5. 初始化动态目标参数（正弦曲线）
        self.target_amplitude = np.array([0.5, 0.8])  # 两个关节的振幅
        self.target_frequency = np.array([0.5, 0.8])  # 两个关节的频率（Hz）
        self.target_offset = np.array([0.0, 0.0])     # 两个关节的偏移量
        self.sim_time = 0.0  # 仿真时间

    def init_data_recording(self):
        """初始化数据记录变量"""
        self.time_history = []          # 时间序列
        self.qpos_history = []          # 实际位置轨迹
        self.qvel_history = []          # 实际速度轨迹
        self.qref_history = []          # 目标位置轨迹
        self.error_history = []         # 跟踪误差轨迹

    def set_controller_param(self, ctrl_mode, kp, kv):
        if ctrl_mode == "trq_mode":
            self.ctrl_mode = ctrl_mode
            self.model.actuator_gainprm[0, 0] = 0
            self.model.actuator_biasprm[0, 1] = 0
            self.model.actuator_gainprm[1, 0] = 0
            self.model.actuator_biasprm[1, 2] = 0
            self.model.actuator_gainprm[2, 0] = 0
            self.model.actuator_biasprm[2, 1] = 0
    
    def generate_dynamic_target(self, t):
        """生成正弦曲线动态目标位置"""
        qref = self.target_offset + self.target_amplitude * np.sin(2 * np.pi * self.target_frequency * t)
        return qref

    def controller(self, model, data):
        """
            Model-Based Controller with Dynamic Target
                (1) PID Controller:
                    tau = Kp * (q_ref - q) - Kd * dq + Ki * ∫(q_ref - q) dt
                (2) PID Controller + Feedforward comp(gravity + coriolis forces):
                    tau = Kp * (q_ref - q) - Kd * dq + Ki * ∫(q_ref - q) dt + C + G
                (3) Feedback linearization with PID:
                    tau = M * (Kp * (q_ref - q) - Kd * dq + Ki * ∫(q_ref - q) dt) + C + G
        """
        if not self.running:
            return
            
        # 生成动态目标位置（正弦曲线）
        qref = self.generate_dynamic_target(self.sim_time)
        # 1. 获取动力学参数
        M = np.zeros((2, 2))
        mj.mj_fullM(model, M, data.qM)
        f = data.qfrc_bias
        # 3. 计算位置误差
        pos_error = qref - data.qpos
        # 4. 更新积分误差（带限幅防止积分饱和）
        # 积分分离优化示例
        error_threshold = 0.5
        if np.linalg.norm(pos_error) < error_threshold:
            self.integral_error += pos_error * model.opt.timestep
        else:
            self.integral_error *= 0.9  # 误差大时减小积分项
        # 积分饱和限制
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        # 5. 计算PID控制项
        proportional = self.Kp @ pos_error
        derivative = -self.Kd @ data.qvel
        integral = self.Ki @ self.integral_error
        ctrl1 = proportional + derivative + integral
        ctrl2 = f
        ctrl3 = M @ ctrl1
        # 6. 设置控制器（选择需要的控制策略）
        ## PID Controller
        # data.qfrc_applied = ctrl1
        ## PID + Feedforward
        # data.qfrc_applied = ctrl1 + ctrl2
        ## PID + Feedback Linearization
        data.qfrc_applied = ctrl3 + ctrl2
        
        # 7. 记录当前数据
        self.record_data(qref, pos_error)
        # 8. 更新仿真时间
        self.sim_time += model.opt.timestep
        self.robot_fk()
        self.robot_ik()

    def record_data(self, qref, pos_error):
        """记录当前时刻的仿真数据"""
        self.time_history.append(self.sim_time)
        self.qpos_history.append(self.data.qpos.copy())
        self.qvel_history.append(self.data.qvel.copy())
        self.qref_history.append(qref.copy())
        self.error_history.append(pos_error.copy())

    def plot_results(self):
        """绘制轨迹跟踪和误差曲线"""
        if not self.time_history:
            return
            
        # 转换为numpy数组
        time_array = np.array(self.time_history)
        qpos_array = np.array(self.qpos_history)
        qref_array = np.array(self.qref_history)
        error_array = np.array(self.error_history)
        
        # 创建绘图窗口
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 绘制轨迹跟踪曲线
        ax1.plot(time_array, qref_array[:, 0], 'r--', linewidth=2, label='Joint 1 Target')
        ax1.plot(time_array, qpos_array[:, 0], 'r-', linewidth=1.5, label='Joint 1 Actual')
        ax1.plot(time_array, qref_array[:, 1], 'b--', linewidth=2, label='Joint 2 Target')
        ax1.plot(time_array, qpos_array[:, 1], 'b-', linewidth=1.5, label='Joint 2 Actual')
        ax1.set_ylabel('Position (rad)')
        ax1.set_title('Double Pendulum Trajectory Tracking')
        ax1.grid(True)
        ax1.legend()
        
        # 绘制跟踪误差曲线
        ax2.plot(time_array, error_array[:, 0], 'r-', linewidth=1.5, label='Joint 1 Error')
        ax2.plot(time_array, error_array[:, 1], 'b-', linewidth=1.5, label='Joint 2 Error')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (rad)')
        ax2.set_title('Tracking Error')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('trajectory_tracking_results.png')  # 保存图片
        plt.show()  # 显示图片

    def main(self):
        sim_start, sim_end = time.time(), 10.0
        while time.time() - sim_start < sim_end and self.running:
            step_start = time.time()
            loop_num, loop_count = 50, 0
            # 1. running for 0.002*50 = 0.1s
            while loop_count < loop_num and self.running:
                loop_count = loop_count + 1
                mj.mj_step(self.model, self.data)
            # 2. GUI show
            if self.is_show:
                if self.viewer.is_running():
                    # self.viewer.cam.lookat[0] = self.data.qpos[0]
                    self.viewer.sync()
                else:
                    break
            # 3. sleep for next period (使用短时间睡眠并轮询)
            step_next_delta = self.model.opt.timestep * loop_num - (time.time() - step_start)
            sleep_time = 0.0
            while self.running and sleep_time < step_next_delta:
                time.sleep(0.001)  # 短时间睡眠，确保能响应中断
                sleep_time += 0.001
                
        # 4. 确保Viewer正确关闭
        if self.is_show and hasattr(self, 'viewer'):
            self.viewer.close()
            
        # 5. 如果有数据，绘制结果
        if self.time_history:
            self.plot_results()


    def keyboard_cb(self, keycode):
        if chr(keycode) == ' ':
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.init_controller()
            # 重置数据记录
            self.init_data_recording()
            self.sim_time = 0.0

    def robot_fk(self):
        """
            x = l1 * cos(q1) + l2 * cos(q1 + q2)
            y = l1 * sin(q1) + l2 * sin(q1 + q2)
        """
        q1 = self.data.qpos[0]
        q2 = self.data.qpos[1]
        l1, l2 = 1, 1
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        
        # 修正传感器索引：framepos在sensordata中的位置是4-6（索引从0开始）
        site_x = self.data.sensordata[4]  # ee_x
        site_y = self.data.sensordata[5]  # ee_y
        
        print("------------- fk ---------------")
        print("fk comp: ", x, y)
        print("sensor: ", site_x, site_y)


    def robot_ik(self):
        """
            r = sqrt(x^2 + y^2)
            cos(q2a) = (l1^2 + l2^2 - r^2) / (2 * l1 * l2)
            q2 = pi - q2a or -pi + q2a
            sin(q1a) = sin(q2a) / r * l2
            q1 = atan(y, x) - q1a
        """
        # 修正传感器索引：framepos在sensordata中的位置是4-6（索引从0开始）
        site_x = self.data.sensordata[4]  # ee_x
        site_y = self.data.sensordata[5]  # ee_y
        
        sensor_q1 = self.data.qpos[0]
        sensor_q2 = self.data.qpos[1]
        l1, l2 = 1, 1
        
        # 添加错误检查，避免除零和数学域错误
        r = np.sqrt(site_x * site_x + site_y * site_y)
        
        # 检查r是否为0或太小
        if r < 1e-6:
            print("Warning: End effector position is too close to origin, r =", r)
            return
            
        # 计算cosq2a并确保其在[-1, 1]范围内
        cosq2a_numerator = (l1 * l1 + l2 * l2 - r * r)
        cosq2a_denominator = (2 * l1 * l2)
        
        # 检查分母是否为0
        if cosq2a_denominator == 0:
            print("Warning: Denominator for cosq2a is zero")
            return
            
        cosq2a = cosq2a_numerator / cosq2a_denominator
        cosq2a = min(1.0, max(-1.0, cosq2a))  # 确保在[-1, 1]范围内
        
        # 计算两个可能的q2值
        try:
            q2a = math.acos(cosq2a)
            q2a1 = q2a - math.pi
            q2a2 = math.pi - q2a
            
            # 选择接近当前q2的解
            q2 = q2a1
            if math.fabs(q2a1 - sensor_q2) > math.fabs(q2a2 - sensor_q2):
                q2 = q2a2
                
            # 计算sinq1a并确保其在[-1, 1]范围内
            sinq1a_numerator = math.sin(math.pi - q2) * l2
            sinq1a = sinq1a_numerator / r
            sinq1a = min(1.0, max(-1.0, sinq1a))  # 确保在[-1, 1]范围内
            
            # 计算q1
            q1 = math.atan2(site_y, site_x) - math.asin(sinq1a)
            
            print("------------- ik ---------------")
            print("ik comp: ", q1, q2)
            print("sensor: ", sensor_q1, sensor_q2)
        except Exception as e:
            print(f"Warning: IK calculation error: {e}")
            print(f"  site_x, site_y: {site_x}, {site_y}")
            print(f"  r: {r}")
            print(f"  cosq2a: {cosq2a}")
            return


if __name__ == "__main__":
    xml_path = "/home/user/workspace/mujoco_ws/env/double_pendulum.xml"
    is_show = True
    Control = Double_Pendulum_Control(xml_path, is_show)
    Control.main()