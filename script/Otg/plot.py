import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# 读取txt文件数据并绘制轨迹和速度曲线

def read_trajectory_data(file_path):
    """
    读取轨迹数据文件
    每行包含14个数据：前7个是关节位置，后7个是关节速度
    """
    positions = []
    velocities = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # 分割每行的数据
            data = line.strip().split()
            if len(data) == 14:
                # 前7个是关节位置
                joint_pos = [float(x) for x in data[:7]]
                # 后7个是关节速度
                joint_vel = [float(x) for x in data[7:]]
                
                positions.append(joint_pos)
                velocities.append(joint_vel)
    
    return np.array(positions), np.array(velocities)

def plot_joint_trajectories(positions, velocities, dt, N):
    """
    绘制关节位置和速度曲线
    """
    # 创建时间轴（假设数据点是等时间间隔的）
    time = np.arange(len(positions)) * dt / (1000 * N)  # 时间点：0, 1, 2, ..., len(traj)-1）
    
    # 创建图形和子图布局
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 1])
    
    # 关节位置图
    ax1 = fig.add_subplot(gs[0])
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
    
    for i in range(7):
        ax1.plot(time, positions[:, i], color=colors[i], label=joint_names[i], linewidth=2)
    
    ax1.set_title('Joint Positions vs Time', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Position (rad)', fontsize=12)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax1.grid(True, alpha=0.3)
    
    # 关节速度图
    ax2 = fig.add_subplot(gs[1])
    for i in range(7):
        ax2.plot(time, velocities[:, i], color=colors[i], label=joint_names[i], linewidth=2)
    
    ax2.set_title('Joint Velocities vs Time', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Velocity (rad/s)', fontsize=12)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 单独显示每个关节的位置和速度对比
    plot_individual_joints(positions, velocities, time)


def plot_all_trajectories(command, positions, velocities, dt, N, delay_count, show_range):
    """
    绘制下发指令、规划轨迹关节位置和速度曲线
    """
    # 创建时间轴（假设数据点是等时间间隔的）
    time_command = (np.arange(len(command))  + delay_count) * dt / (1000)
    time = np.arange(len(positions)) * dt / (1000 * N)  # 时间点：0, 1, 2, ..., len(traj)-1）
    
    # 创建图形和子图布局
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 1])

    if show_range == []:
        show_range = [0, time[-1]]
    # 关节位置图
    ax1 = fig.add_subplot(gs[0])
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
    
    for i in range(7):
        ax1.plot(time, positions[:, i], color=colors[i], label=joint_names[i], linewidth=2)
        ax1.plot(time_command, command[:, i], color=colors[i], label=joint_names[i], linewidth=2, linestyle='--')

    ax1.set_title('Joint Positions vs Time', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Position (rad)', fontsize=12)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax1.grid(True, alpha=0.3)
    plt.xlim(show_range[0], show_range[1])      # 设置x轴范围

    
    # 关节速度图
    ax2 = fig.add_subplot(gs[1])
    for i in range(7):
        ax2.plot(time, velocities[:, i], color=colors[i], label=joint_names[i], linewidth=2)
    
    ax2.set_title('Joint Velocities vs Time', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Velocity (rad/s)', fontsize=12)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax2.grid(True, alpha=0.3)
    plt.xlim(show_range[0], show_range[1])      # 设置x轴范围
    plt.tight_layout()
    plt.show()
    
    # 单独显示每个关节的位置和速度对比
    plot_individual_all_joints(command, positions, velocities, time_command, time, show_range)

def plot_individual_joints(positions, velocities, time, show_range):
    """
    为每个关节单独绘制位置和速度对比图
    """
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(7):
        ax = axes[i]
        
        # 绘制位置曲线（左侧y轴）
        ax_pos = ax
        line1 = ax_pos.plot(time, positions[:, i], color=colors[i], linewidth=2, label='Position')
        ax_pos.set_ylabel('Position (rad)', color=colors[i], fontsize=10)
        ax_pos.tick_params(axis='y', labelcolor=colors[i])
        
        # 创建右侧y轴用于速度
        ax_vel = ax_pos.twinx()
        line2 = ax_vel.plot(time, velocities[:, i], color='black', linewidth=1, linestyle='--', label='Velocity')
        ax_vel.set_ylabel('Velocity (rad/s)', color='black', fontsize=10)
        ax_vel.tick_params(axis='y', labelcolor='black')
        
        ax_pos.set_title(f'{joint_names[i]} - Position and Velocity', fontsize=12, fontweight='bold')
        ax_pos.set_xlabel('Time Step', fontsize=10)
        ax_pos.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_pos.legend(lines, labels, loc='upper right')
        ax.set_xlim(show_range[0], show_range[1])      # 设置x轴范围

    
    # 隐藏最后一个空子图
    axes[7].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_individual_all_joints(command, positions, velocities, time_command, time, show_range):
    """
    为每个关节单独绘制位置和速度对比图
    """
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(7):
        ax = axes[i]
        
        # 绘制位置曲线（左侧y轴）
        ax_pos = ax
        line1 = ax_pos.plot(time, positions[:, i], color=colors[i], linewidth=2, label='Position')
        line1 += ax_pos.plot(time_command, command[:, i], color=colors[i], linewidth=2, linestyle='--', label='Command')
        ax_pos.set_ylabel('Position (rad)', color=colors[i], fontsize=10)
        ax_pos.tick_params(axis='y', labelcolor=colors[i])
        
        # 创建右侧y轴用于速度
        ax_vel = ax_pos.twinx()
        line2 = ax_vel.plot(time, velocities[:, i], color='black', linewidth=1, linestyle='--', label='Velocity')
        ax_vel.set_ylabel('Velocity (rad/s)', color='black', fontsize=10)
        ax_vel.tick_params(axis='y', labelcolor='black')
        
        ax_pos.set_title(f'{joint_names[i]} - Position and Velocity', fontsize=12, fontweight='bold')
        ax_pos.set_xlabel('Time Step', fontsize=10)
        ax_pos.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_pos.legend(lines, labels, loc='upper right')
        ax.set_xlim(show_range[0], show_range[1])      # 设置x轴范围
    
    # 隐藏最后一个空子图
    axes[7].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def analyze_trajectory_data(positions, velocities):
    """
    分析轨迹数据的统计信息
    """
    print("=== 轨迹数据分析 ===")
    print(f"数据点数: {len(positions)}")
    print(f"关节数量: {positions.shape[1]}")
    print()
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
    
    print("关节位置统计 (rad):")
    print("关节\t\t最小值\t\t最大值\t\t平均值\t\t标准差")
    print("-" * 70)
    for i in range(7):
        min_val = np.min(positions[:, i])
        max_val = np.max(positions[:, i])
        mean_val = np.mean(positions[:, i])
        std_val = np.std(positions[:, i])
        print(f"{joint_names[i]}\t{min_val:.6f}\t{max_val:.6f}\t{mean_val:.6f}\t{std_val:.6f}")
    
    print("\n关节速度统计 (rad/s):")
    print("关节\t\t最小值\t\t最大值\t\t平均值\t\t标准差")
    print("-" * 70)
    for i in range(7):
        min_val = np.min(velocities[:, i])
        max_val = np.max(velocities[:, i])
        mean_val = np.mean(velocities[:, i])
        std_val = np.std(velocities[:, i])
        print(f"{joint_names[i]}\t{min_val:.6f}\t{max_val:.6f}\t{mean_val:.6f}\t{std_val:.6f}")

def main():
    """
    主函数
    """
    # 文件路径
    file_path = "trajData_wk-1-1-5.txt"
    
    try:
        # 读取数据
        print("正在读取轨迹数据...")
        positions, velocities = read_trajectory_data(file_path)
        
        if len(positions) == 0:
            print("错误：未读取到有效数据")
            return
        
        print(f"成功读取 {len(positions)} 个数据点")
        
        # 分析数据
        analyze_trajectory_data(positions, velocities)
        
        # 绘制图形
        print("\n正在生成图形...")
        plot_joint_trajectories(positions, velocities, dt=50, N=1)
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"错误：{e}")

if __name__ == "__main__":
    main()