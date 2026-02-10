import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/util")
from plotter import DataCollector
import numpy as np

# 多轴四阶LTD位置规划器
class MultiAxisConstrainedPositionPlanner:
    def __init__(self, num_axes, omega_c, sample_time, initial_positions=None, 
                 max_velocity=None, max_acceleration=None, max_jerk=None):
        """
        初始化带约束的多轴四阶LTD位置规划器
        
        参数:
            num_axes: 轴的数量
            omega_c: 系统带宽 (rad/s)，可以是标量或长度为num_axes的数组
            sample_time: 采样时间 (s)
            initial_positions: 初始位置数组，长度为num_axes
            max_velocity: 最大速度限制，可以是标量或长度为num_axes的数组
            max_acceleration: 最大加速度限制，可以是标量或长度为num_axes的数组
            max_jerk: 最大加加速度限制，可以是标量或长度为num_axes的数组
        """
        self.num_axes = num_axes
        self.sample_time = sample_time
        
        # 处理参数为数组形式
        self.omega_c = np.array(omega_c) if hasattr(omega_c, '__len__') else np.full(num_axes, omega_c)
        
        # 计算特征多项式系数（每个轴独立）
        self.a0 = self.omega_c ** 4
        self.a1 = 4 * self.omega_c ** 3
        self.a2 = 6 * self.omega_c ** 2
        self.a3 = 4 * self.omega_c
        
        # 初始化状态（每个轴独立）
        if initial_positions is None:
            initial_positions = np.zeros(num_axes)
        self.x1 = np.array(initial_positions, dtype=float)  # 位置
        self.x2 = np.zeros(num_axes)  # 速度
        self.x3 = np.zeros(num_axes)  # 加速度
        self.x4 = np.zeros(num_axes)  # 加加速度
        
        # 设置约束限制
        self.max_velocity = np.array(max_velocity) if hasattr(max_velocity, '__len__') else np.full(num_axes, max_velocity)
        self.max_acceleration = np.array(max_acceleration) if hasattr(max_acceleration, '__len__') else np.full(num_axes, max_acceleration)
        self.max_jerk = np.array(max_jerk) if hasattr(max_jerk, '__len__') else np.full(num_axes, max_jerk)
        
    def _constrain(self, values, limits):
        """辅助方法：对值数组进行限幅"""
        if limits is not None:
            return np.clip(values, -limits, limits)
        return values
        
    def update(self, target_positions):
        """
        更新多轴规划器
        :param target_positions: 目标位置数组，长度为num_axes
        :return: 平滑位置x1，平滑速度x2，平滑加速度x3，平滑加加速度x4
        """
        target_positions = np.array(target_positions)
        
        # 计算无约束的加加速度（每个轴独立）
        # unconstrained_jerk = -self.a0 * (self.x1 - target_positions) \
        #                      - self.a1 * self.x2 \
        #                      - self.a2 * self.x3 \
        #                      - self.a3 * self.x4
        
        # # 状态更新（每个轴独立）
        # self.x1 = self.x1 + self.sample_time * self.x2
        # self.x2 = self.x2 + self.sample_time * self.x3
        # self.x3 = self.x3 + self.sample_time * self.x4
        # self.x4 = self.x4 + self.sample_time * unconstrained_jerk

# 使用梯形积分，更加稳定
        # 计算当前加加速度
        x4_current = -self.a0*(self.x1 - target_positions) - self.a1*self.x2 - self.a2*self.x3 - self.a3*self.x4
        # 梯形积分：(当前值+下一步值)/2 * dt
        self.x3 += (self.x4 + x4_current) / 2 * self.sample_time
        self.x2 += (self.x3 + (self.x3 + x4_current*self.sample_time)) / 2 * self.sample_time
        self.x1 += (self.x2 + (self.x2 + self.x3*self.sample_time)) / 2 * self.sample_time
        self.x4 = x4_current - 0.1*self.x4  # 更新加加速度 添加阻尼项

        # 约束处理（每个轴独立）
        self.x2 = self._constrain(self.x2, self.max_velocity)
        self.x3 = self._constrain(self.x3, self.max_acceleration)
        self.x4 = self._constrain(self.x4, self.max_jerk)

        return self.x1.copy(), self.x2.copy(), self.x3.copy(), self.x4.copy()

    def get_states(self):
        """获取当前所有轴的状态"""
        return {
            'positions': self.x1.copy(),
            'velocities': self.x2.copy(),
            'accelerations': self.x3.copy(),
            'jerks': self.x4.copy()
        }

# 多轴一阶跟踪微分器
class MultiAxisFirstOrderTD:
    def __init__(self, num_axes, initial_positions=None, k=10, dt=0.001):
        """
        初始化多轴一阶跟踪微分器
        :param num_axes: 轴的数量
        :param k: 跟踪系数，可以是标量或长度为num_axes的数组
        :param dt: 控制周期/采样时间（秒）
        """
        self.num_axes = num_axes
        self.dt = dt
        
        # 处理跟踪系数为数组形式
        self.k = np.array(k) if hasattr(k, '__len__') else np.full(num_axes, k)
        
        # 初始化状态（每个轴独立）
        if initial_positions is None:
            self.x1 = np.zeros(num_axes)  # 跟踪位置
        else:
            self.x1 = np.array(initial_positions, dtype=float)  # 跟踪位置
        self.x2 = np.zeros(num_axes)  # 跟踪速度

    def update(self, targets):
        """
        更新多轴跟踪微分器
        :param targets: 目标位置数组，长度为num_axes
        :return: 平滑位置x1，平滑速度x2
        """
        targets = np.array(targets)
        
        # 核心：多轴TD的位置更新公式（每个轴独立）
        self.x1 = self.x1 + self.k * (targets - self.x1) * self.dt
        # 核心：多轴TD的速度（微分）计算（每个轴独立）
        self.x2 = self.k * (targets - self.x1)
        
        return self.x1.copy(), self.x2.copy()
    
    def get_states(self):
        """获取当前所有轴的状态"""
        return {
            'positions': self.x1.copy(),
            'velocities': self.x2.copy()
        }

# -------------------------- 2. 7DoF Robotic Arm Signal Simulation --------------------------
def simulate_7dof_arm(t_total=5, h=0.001):
    """
    Simulate 7DoF robotic arm joint position signals (with noise)
    t_total: Total simulation time (seconds)
    h: Sampling step (seconds)
    Returns:
        t: Time sequence
        joint_pos_raw: 7 joint raw noisy positions (shape: (7, N))
        joint_pos_true: 7 joint true noise-free positions (shape: (7, N))
    """
    # Generate time sequence
    t = np.arange(0, t_total, h)
    N = len(t)
    
    # 7 joint true motion trajectories (sine curves with different frequencies, simulating actual motion)
    joint_freq = [0.5, 0.7, 1.0, 0.8, 1.2, 0.6, 0.9]  # Motion frequency for each joint
    joint_amp = [10, 20, 18, 22, 16, 19, 17]         # Motion amplitude for each joint (°)
    joint_pos_true = np.zeros((7, N))
    
    for i in range(7):
        # Sine trajectory + small offset, simulating joint motion
        joint_pos_true[i] = joint_amp[i] * np.sin(2 * np.pi * joint_freq[i] * t) + 5
    
    # Add sensor noise (Gaussian white noise, simulating encoder noise)
    noise = np.random.normal(0, 0.3, (7, N))  # Noise amplitude ±0.3° (typical industrial encoder noise)
    joint_pos_raw = joint_pos_true + noise
    
    return t, joint_pos_raw, joint_pos_true

# 示例使用函数
def demo_multi_axis_td():
    """演示多轴跟踪微分器的使用"""
    num_joints = 7  # 7个关节
    
    # 模拟数据
    total_time = 5
    dt = 0.002
    h = 0.05
    
    # 读取轨迹数据
    from trajectoryPlan import read_trajectory_data
    file_path = "trajData_wk-1-1-5.txt"
    positions, velocities = read_trajectory_data(file_path)
    if len(positions) == 0:
        print("错误：未读取到有效数据")
        return
    
    joint_pos_raw = positions.T  # 转换为(7, N)形状
    
    # 优化：设置所有关节的初始位置为轨迹的第一个点
    initial_positions_all = joint_pos_raw[:, 0]  # 获取所有关节的第一个位置点
    
    # 创建多轴跟踪微分器（每个关节可以有不同的参数）
    td_planner = MultiAxisFirstOrderTD(
        num_axes=num_joints,
        k=[10, 12, 8, 15, 9, 11, 13],  # 每个关节不同的跟踪系数
        dt=0.002
    )
    
    # 创建多轴四阶规划器，使用轨迹的第一个点作为初始位置
    ltd_planner = MultiAxisConstrainedPositionPlanner(
        num_axes=num_joints,
        omega_c=[1.0, 1.2, 0.8, 1.5, 0.9, 1.1, 1.3],  # 每个关节不同的带宽
        sample_time=0.1,
        initial_positions=initial_positions_all,  # 使用所有关节的初始位置
        max_velocity=1.0,
        max_acceleration=1.0,
        max_jerk=1.0
    )
    
    # 创建数据收集器（每个关节一个）
    plotters_td = [DataCollector() for _ in range(num_joints)]
    plotters_ltd = [DataCollector() for _ in range(num_joints)]
    
    t_list = np.arange(0, total_time, dt)
    
    for t in t_list:
        # 获取当前时间所有关节的目标位置
        time_index = int(t / h)
        if time_index >= joint_pos_raw.shape[1]:
            break
            
        target_positions = joint_pos_raw[:, time_index]
        
        # 更新跟踪微分器
        td_positions, td_velocities = td_planner.update(target_positions)
        
        # 更新四阶规划器
        ltd_positions, ltd_velocities, ltd_accelerations, ltd_jerks = ltd_planner.update(target_positions)
        
        # 收集每个关节的数据
        for joint_idx in range(num_joints):
            plotters_td[joint_idx].add_data(t, td_positions[joint_idx], target_positions[joint_idx])
            plotters_td[joint_idx].add_vel(td_velocities[joint_idx])
            
            plotters_ltd[joint_idx].add_data(t, ltd_positions[joint_idx], target_positions[joint_idx])
            plotters_ltd[joint_idx].add_vel(ltd_velocities[joint_idx])
    
    # 绘制每个关节的结果
    for joint_idx in range(num_joints):
        plotters_td[joint_idx].plot_single_axis(
            0, 
            save_to_file=True, 
            filename=f"joint_{joint_idx+1}_td.png"
        )
        plotters_ltd[joint_idx].plot_single_axis(
            0, 
            save_to_file=True, 
            filename=f"joint_{joint_idx+1}_ltd.png"
        )
    
    print("多轴跟踪微分器演示完成！")

# ... existing code ...
if __name__ == "__main__":
    # 原来的单轴演示代码（保持兼容性）
    plotter = DataCollector()
    plotter1 = DataCollector()

    # 1. Simulate 7DoF robotic arm signals
    total_time = 5
    dt = 0.002
    h = 0.05  # 1ms sampling period (typical industrial robot controller value)
    
    from trajectoryPlan import read_trajectory_data
    file_path = "trajData_wk-1-1-5.txt"
    positions, velocities = read_trajectory_data(file_path)
    if len(positions) == 0:
        print("错误：未读取到有效数据")
        exit(1)
    print(f"成功读取 {len(positions)} 个数据点")
    joint_pos_raw = positions.T
    
    # 优化：设置所有关节的初始位置为轨迹的第一个点
    initial_positions_all = joint_pos_raw[:, 0]  # 获取所有关节的第一个位置点
    print(f"初始位置: {initial_positions_all}")
    # 创建单轴四阶LTD位置规划器（保持向后兼容）
    planner = MultiAxisConstrainedPositionPlanner(
        num_axes=1, 
        omega_c=1, 
        sample_time=0.1, 
        initial_positions=initial_positions_all[0],  # 使用第一个关节的初始位置
        max_velocity=1.0, 
        max_acceleration=1.0, 
        max_jerk=1.0
    )
    
    # 创建单轴一阶跟踪微分器（保持向后兼容）
    first_order_td = MultiAxisFirstOrderTD(num_axes=1, k=10, dt=dt)

    t_list = np.arange(0, total_time, dt)
    for t in t_list:
        time_index = int(t/h)
        if time_index >= joint_pos_raw.shape[1]:
            break
            
        target_pos = joint_pos_raw[0][time_index]
        # target_pos = 1 if t > 2 else 0
        
        # 更新规划器（单轴模式）
        pos_array, vel_array, acc_array, jerk_array = planner.update([target_pos])
        pos1_array, vel1_array = first_order_td.update([target_pos])
        
        pos, vel, acc, jerk = pos_array[0], vel_array[0], acc_array[0], jerk_array[0]
        pos1, vel1 = pos1_array[0], vel1_array[0]
        
        # print(f"时间{t:.4f}: 位置={pos:.4f}, 速度={vel:.4f}, 加速度={acc:.4f}, 加加速度={jerk:.4f}")
        plotter.add_data(t, pos, target_pos)
        plotter.add_vel(vel)
        plotter1.add_data(t, pos1, target_pos)
        plotter1.add_vel(vel1)

    plotter.plot_single_axis(0, save_to_file=True, filename="single_axis.png")
    plotter1.plot_single_axis(0, save_to_file=True, filename="single_axis1.png")
    
    # 运行多轴演示
    print("\n运行多轴跟踪微分器演示...")
    demo_multi_axis_td()