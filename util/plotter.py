import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

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

class DataCollector:
    def __init__(self):
        # 数据收集容器
        self.timestamps = []
        self.actual_pose = []
        self.desired_pose = []
        self.vel = []
        self.idx = ["x", "y", "z", "r", "p", "y"]
        
    def add_data(self, timestamp, actual_pose, desired_pose):
        """收集数据点"""
        self.timestamps.append(timestamp)
        self.actual_pose.append(actual_pose)
        self.desired_pose.append(desired_pose)

    def add_vel(self, vel):
        """收集数据点"""
        self.vel.append(vel)
    
    def plot_after_simulation(self, save_to_file=False, filename="simulation_results.png"):
        """仿真结束后显示绘图界面"""
        if not self.timestamps:
            print("没有数据可绘制")
            return
            
        print("仿真完成，开始绘制图表...")
        
        # 转换为numpy数组便于索引
        actual_pose_array = np.array(self.actual_pose)
        desired_pose_array = np.array(self.desired_pose)
        
        # 创建图形，增加高度以容纳统计信息
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 3, 3, 1])  # 最后一行用于统计信息
        
        fig.suptitle('Robot Control - Simulation Results with Statistics')
        
        # 存储每个轴的误差统计
        axis_stats = []
        
        # 绘制位置和姿态跟踪
        for i, name in enumerate(self.idx):
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])
            
            # 获取当前轴的数据
            actual_data = actual_pose_array[:, i]
            desired_data = desired_pose_array[:, i]
            
            # 绘制数据
            ax.plot(self.timestamps, actual_data, 'r-', label='Actual', linewidth=1)
            ax.plot(self.timestamps, desired_data, 'g-', label='Desired', linewidth=0.5)
            
            # 计算误差统计
            error = np.abs(actual_data - desired_data)
            max_error = np.max(error)
            mean_error = np.mean(error)
            rmse = np.sqrt(np.mean(error**2))
            
            # 存储统计信息
            axis_stats.append({
                'axis': name.upper(),
                'max_error': max_error,
                'mean_error': mean_error,
                'rmse': rmse
            })
            
            # 在子图中添加误差统计
            stats_text = f'Max: {max_error:.4f}\nMean: {mean_error:.4f}\nRMSE: {rmse:.4f}'
            ax.text(0.02, 0.98, stats_text, 
                    transform=ax.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            title = f"{name.upper()} Tracking"
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            if i < 3:
                ax.set_ylabel('Position (m)')
            else:
                ax.set_ylabel('Orientation (rad)')
            ax.legend()
            ax.grid(True)
        
        # 添加全局统计信息表格
        ax_stats = fig.add_subplot(gs[3, :])
        ax_stats.axis('off')
        
        # 创建统计表格
        table_data = []
        for stat in axis_stats:
            table_data.append([
                stat['axis'],
                f"{stat['max_error']:.6f}",
                f"{stat['mean_error']:.6f}",
                f"{stat['rmse']:.6f}"
            ])
        
        # 计算全局统计
        all_errors = np.abs(actual_pose_array - desired_pose_array)
        global_max_error = np.max(all_errors)
        global_mean_error = np.mean(all_errors)
        global_rmse = np.sqrt(np.mean(all_errors**2))
        
        table_data.append([
            'GLOBAL',
            f"{global_max_error:.6f}",
            f"{global_mean_error:.6f}",
            f"{global_rmse:.6f}"
        ])
        
        # 创建表格
        table = ax_stats.table(
            cellText=table_data,
            colLabels=['Axis', 'Max Error', 'Mean Error', 'RMSE'],
            loc='center',
            cellLoc='center'
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 设置全局统计行样式
        table[(len(axis_stats)+1, 0)].set_facecolor('lightgreen')
        table[(len(axis_stats)+1, 1)].set_facecolor('lightgreen')
        table[(len(axis_stats)+1, 2)].set_facecolor('lightgreen')
        table[(len(axis_stats)+1, 3)].set_facecolor('lightgreen')
        
        # 打印统计信息到控制台
        print("\n=== Simulation Statistics ===")
        for stat in axis_stats:
            print(f"{stat['axis']}: Max={stat['max_error']:.6f}, Mean={stat['mean_error']:.6f}, RMSE={stat['rmse']:.6f}")
        print(f"GLOBAL: Max={global_max_error:.6f}, Mean={global_mean_error:.6f}, RMSE={global_rmse:.6f}")
        print("=============================\n")
        
        # 调整布局
        plt.tight_layout()
        
        if save_to_file:
            # 保存到文件，避免显示问题
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {filename}")
        else:
            # 尝试显示图表
            try:
                print("正在显示绘图界面...")
                plt.show()
                print("图表显示完成")
            except Exception as e:
                print(f"显示图表失败: {e}")
                print("尝试保存图表到文件...")
                plt.savefig("simulation_results.png", dpi=300, bbox_inches='tight')
                print("图表已保存到: simulation_results.png")


    def plot_single_axis(self, axis_index, save_to_file=False, filename="single_axis.png"):
        """绘制单个轴的轨迹和速度曲线（适配一维和多维数组）"""
        if not self.timestamps:
            print("没有数据可绘制")
            return
            
        print("仿真完成，开始绘制图表...")
        # 转换为numpy数组便于索引
        actual_pose_array = np.array(self.actual_pose)
        desired_pose_array = np.array(self.desired_pose)
        
        # 检查数组维度并适配索引方式
        if actual_pose_array.ndim == 1:
            # 一维数组情况：直接使用整个数组
            actual_data = actual_pose_array
            desired_data = desired_pose_array
            title_suffix = " (1D Data)"
        elif actual_pose_array.ndim == 2:
            # 二维数组情况：按轴索引
            if axis_index >= actual_pose_array.shape[1]:
                print(f"错误：轴索引 {axis_index} 超出范围（最大 {actual_pose_array.shape[1]-1}）")
                return
            actual_data = actual_pose_array[:, axis_index]
            desired_data = desired_pose_array[:, axis_index]
            title_suffix = f" - Axis {axis_index}"
        else:
            print(f"错误：不支持 {actual_pose_array.ndim} 维数组")
            return
        
        # 检查是否有速度数据
        has_velocity_data = len(self.vel) > 0
        
        # 创建图形：如果有速度数据，创建2个子图；否则创建1个子图
        if has_velocity_data:
            fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f'Robot Control - Position and Velocity Tracking{title_suffix}')
        else:
            fig, ax_pos = plt.subplots(1, 1, figsize=(10, 6))
            fig.suptitle(f'Robot Control - Position Tracking{title_suffix}')
        
        # 绘制位置跟踪曲线
        ax_pos.plot(self.timestamps, desired_data, 'g-', label='Desired Position', linewidth=1, alpha=0.7)
        ax_pos.plot(self.timestamps, actual_data, 'r-', label='Actual Position', linewidth=2)
        
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position')
        ax_pos.legend()
        ax_pos.grid(True)
        
        # 添加位置统计信息
        if len(actual_data) > 0 and len(desired_data) > 0:
            error = np.abs(actual_data - desired_data)
            max_error = np.max(error)
            mean_error = np.mean(error)
            rmse = np.sqrt(np.mean(error**2))
            
            # 在图上添加位置误差统计
            pos_stats_text = f'Position Errors:\nMax: {max_error:.4f}\nMean: {mean_error:.4f}\nRMSE: {rmse:.4f}'
            ax_pos.text(0.02, 0.98, pos_stats_text, 
                       transform=ax_pos.transAxes, verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 如果有速度数据，绘制速度曲线
        if has_velocity_data:
            # 转换为numpy数组便于索引
            vel_array = np.array(self.vel)
            
            # 检查速度数组维度并适配索引方式
            if vel_array.ndim == 1:
                # 一维数组情况：直接使用整个数组
                vel_data = vel_array
            elif vel_array.ndim == 2:
                # 二维数组情况：按轴索引
                if axis_index >= vel_array.shape[1]:
                    print(f"警告：速度数据轴索引 {axis_index} 超出范围，使用第一个轴")
                    vel_data = vel_array[:, 0]
                else:
                    vel_data = vel_array[:, axis_index]
            else:
                print(f"警告：不支持的速度数据维度 {vel_array.ndim}，跳过速度绘图")
                has_velocity_data = False
            
            if has_velocity_data:
                # 绘制速度曲线
                ax_vel.plot(self.timestamps, vel_data, 'b-', label='Velocity', linewidth=2)
                
                # 计算速度统计信息
                vel_max = np.max(np.abs(vel_data))
                vel_mean = np.mean(np.abs(vel_data))
                vel_std = np.std(vel_data)
                
                # 在图上添加速度统计
                vel_stats_text = f'Velocity Stats:\nMax: {vel_max:.4f}\nMean: {vel_mean:.4f}\nStd: {vel_std:.4f}'
                ax_vel.text(0.02, 0.98, vel_stats_text, 
                           transform=ax_vel.transAxes, verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                ax_vel.set_xlabel('Time (s)')
                ax_vel.set_ylabel('Velocity')
                ax_vel.legend()
                ax_vel.grid(True)
                
                # 打印速度统计到控制台
                print(f"速度统计 - 轴 {axis_index}: 最大速度={vel_max:.4f}, 平均速度={vel_mean:.4f}, 标准差={vel_std:.4f}")
        
        # 调整布局
        plt.tight_layout()
        
        if save_to_file:
            # 保存到文件，避免显示问题
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {filename}")
        else:
            # 尝试显示图表
            try:
                print("正在显示绘图界面...")
                plt.show()
                print("图表显示完成")
            except Exception as e:
                print(f"显示图表失败: {e}")
                print("尝试保存图表到文件...")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"图表已保存到: {filename}")

if __name__ == "__main__":
    plotter = DataCollector()
    for i in range(100):
        actual_pose = np.array([np.sin(i*0.1), np.cos(i*0.1), 0.5, 0.1, 0.2, 0.3])
        desired_pose = np.array([np.sin(i*0.1+0.1), np.cos(i*0.1+0.1), 0.5, 0.1, 0.2, 0.3])
        plotter.add_data(i*0.01, actual_pose, desired_pose)
    
    # 使用保存到文件模式，避免显示问题
    plotter.plot_after_simulation(save_to_file=False)
    # plotter.plot_single_axis(1, save_to_file=True)