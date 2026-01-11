import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

class DataCollector:
    def __init__(self):
        # 数据收集容器
        self.timestamps = []
        self.actual_pose = []
        self.desired_pose = []
        self.idx = ["x", "y", "z", "r", "p", "y"]
        
    def add_data(self, timestamp, actual_pose, desired_pose):
        """收集数据点"""
        self.timestamps.append(timestamp)
        self.actual_pose.append(actual_pose)
        self.desired_pose.append(desired_pose)
    
    def plot_after_simulation(self, save_to_file=False, filename="simulation_results.png"):
        """仿真结束后显示绘图界面"""
        if not self.timestamps:
            print("没有数据可绘制")
            return
            
        print("仿真完成，开始绘制图表...")
        
        # 转换为numpy数组便于索引
        actual_pose_array = np.array(self.actual_pose)
        desired_pose_array = np.array(self.desired_pose)
        
        # 创建图形
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle('Robot Control - Simulation Results')
        
        # 绘制位置和姿态跟踪
        for i, name in enumerate(self.idx):
            row = i // 2
            col = i % 2
            ax = axs[row, col]
            
            ax.plot(self.timestamps, actual_pose_array[:, i], 'b-', label='Actual', linewidth=2)
            ax.plot(self.timestamps, desired_pose_array[:, i], 'r--', label='Desired', linewidth=2)
            
            title = f"{name.upper()} Tracking"
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            if i < 3:
                ax.set_ylabel('Position (m)')
            else:
                ax.set_ylabel('Orientation (rad)')
            ax.legend()
            ax.grid(True)
        
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

if __name__ == "__main__":
    plotter = DataCollector()
    for i in range(100):
        actual_pose = np.array([np.sin(i*0.1), np.cos(i*0.1), 0.5, 0.1, 0.2, 0.3])
        desired_pose = np.array([np.sin(i*0.1+0.1), np.cos(i*0.1+0.1), 0.5, 0.1, 0.2, 0.3])
        plotter.add_data(i*0.01, actual_pose, desired_pose)
    
    # 使用保存到文件模式，避免显示问题
    plotter.plot_after_simulation(save_to_file=False)