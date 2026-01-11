import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
from plot import read_trajectory_data, plot_all_trajectories
# 测试三次多项式轨迹规划
# 轨迹规划+画图
def plot_trajectory(traj, velocity, dt, N):
    """
    绘制轨迹的位移、速度和加速度曲线图
    
    参数:
    traj: np.array类型的轨迹数据，每行包含n个元素，所有元素都是轨迹变量
    
    返回:
    None
    """
    # 设置numpy输出格式：3位浮点数，不使用科学计数法
    np.set_printoptions(precision=3, suppress=True)
    
    # 确保输入是numpy数组
    traj = np.array(traj)
    
    # 获取轨迹维度
    dim = traj.shape[1]
    
    # 生成时间序列（假设每行对应一个时间点，间隔为1秒）
    t = np.arange(len(traj)) * dt / (1000 * N)  # 时间点：0, 1, 2, ..., len(traj)-1）
    
    # 提取位移数据
    displacement = traj[:, 0:dim]

    velocity = velocity[:, 0:dim]
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 绘制位移曲线
    plt.subplot(3, 1, 1)
    for i in range(dim):
        plt.plot(t, displacement[:, i],  linewidth=2, label=f'Var {i+1}')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Displacement', fontsize=12)
    plt.title('Displacement Curves', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 绘制速度曲线
    plt.subplot(3, 1, 2)
    for i in range(dim):
        plt.plot(t, velocity[:, i],  linewidth=2, label=f'Var {i+1}')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Velocity', fontsize=12)
    plt.title('Velocity Curves', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()  # 调整子图间距
    plt.show()
    

from copy import copy
from ruckig import InputParameter, OutputParameter, Result, Ruckig

import numpy as np
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

Flt = np.float32
EPS = 1e-6

def arr(*l):
    return np.array(l, dtype=Flt)

def test_ruckig(t_samples, x_samples, dt):
    # 设置初始条件 - 与 test_package.py 相同
    x0 = arr(x_samples[0])
    v0 = arr(0)
    v_max = arr(3.0)
    a_max = arr(20.0)
    j_max = arr(330)
    fps = 1000. / dt
    
    # 创建 Ruckig 实例
    ruckig = Ruckig(1, 1.0/fps)  # 1维，周期1/fps
    
    # 设置输入参数
    inp = InputParameter(1)
    inp.current_position = x0
    inp.current_velocity = v0
    inp.current_acceleration = arr(0)
    
    # 设置运动限制
    inp.max_velocity = v_max
    inp.max_acceleration = a_max
    inp.max_jerk = j_max
    
    # 目标轨迹 - 与 test_package.py 相同
    v_samples = np.zeros_like(x_samples)
    a_samples = np.zeros_like(x_samples)
    
    # 生成轨迹
    res = []
    t_vals = []
    out = OutputParameter(1)
    print(f"x_samples {x_samples.shape}")
    last_itpltn_t = 0
    for i, (x_target, v_target, a_target, t_target) in enumerate(zip(x_samples, v_samples, a_samples, t_samples)):
        points_needed = int((t_target - last_itpltn_t) * fps)
        print(f"i {i} t_target {t_target} points_needed {points_needed}")
        # 设置目标状态
        inp.target_position = arr(x_target)
        if i < len(x_samples) - 1:
            v_target = arr(x_samples[i + 1] - x_samples[i - 1]/(dt*2))
        else:
            v_target = arr(0)
        # inp.target_velocity = arr(v_target)
        v_samples[i] = v_target
        # inp.target_acceleration = arr(a_target)
        
        # 计算轨迹
        result = Result.Working
        local_t_vals = []
        local_res = []
        
        current_time = 0.0
        time_step = 1.0 / fps
        prev_acceleration = float(inp.current_acceleration[0])
        
        for point_idx in range(points_needed):
            result = ruckig.update(inp, out)
            
            if result == Result.Working or result == Result.Finished:
                # 计算急动度 (jerk) - 加速度的变化率
                current_acceleration = float(out.new_acceleration[0])
                jerk = (current_acceleration - prev_acceleration) / time_step
                
                local_res.append([
                    float(out.new_position[0]),
                    float(out.new_velocity[0]), 
                    current_acceleration,
                    jerk
                ])
                local_t_vals.append(last_itpltn_t + point_idx * time_step)
                
                # 更新当前状态为新的状态
                inp.current_position = out.new_position
                inp.current_velocity = out.new_velocity
                prev_acceleration = current_acceleration
                inp.current_acceleration = out.new_acceleration
            else:
                print(f"Ruckig failed at point {point_idx}: {result}")
                break
        
        res.extend(local_res)
        t_vals.extend(local_t_vals)
        last_itpltn_t += points_needed / fps
    
    res = np.array(res, dtype=np.float32)
    x_vals = res[:, 0]
    v_vals = res[:, 1]
    a_vals = res[:, 2]
    j_vals = res[:, 3]
    print(f"x_vals {len(x_vals)}")

    def plot_subplots(t, data, titles, y_labels, extra_data=None, extra_t=None):
        print(f"plot_subplots  {len(data[0])}")
        # 设置字体和样式 - 与 test_package.py 相同
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 8,                 
            'font.family': 'sans-serif',  # 改为sans-serif以支持中文
            'mathtext.fontset': 'stix',
            'axes.labelsize': 9,           
            'axes.titlesize': 10,          
            'xtick.labelsize': 8,          
            'ytick.labelsize': 8,
            'legend.fontsize': 7,          
            'figure.dpi': 300,
            'lines.linewidth': 1.0,        
            'lines.markersize': 3,         
            'axes.linewidth': 0.8,         
            'grid.linewidth': 0.4,         
        })

        fig, axs = plt.subplots(4, 1, figsize=(5, 6))
        fig.subplots_adjust(hspace=0.60)

        for i, (ax, data_vals, title, y_label) in enumerate(zip(axs, data, titles, y_labels)):
            for j in range(data_vals.shape[1]):
                ax.plot(
                    t,
                    data_vals[:, j],
                    color='black',
                    marker='.',
                    markersize=2,
                    linestyle='-',
                    zorder=3,
                )
            
            if extra_data is not None and extra_t is not None and i < len(extra_data):
                ax.plot(
                    extra_t,
                    extra_data[i],
                    'x',
                    color='#FF2C00',
                    markersize=3,
                    linestyle='--',
                    zorder=2,
                )

            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.4, zorder=1)
            
            # 设置刻度
            ax.tick_params(direction='in', length=2, width=0.8, colors='black',
                        grid_color='gray', grid_alpha=0.5)
            
            # 设置边框
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            
            # 使用支持中文的字体设置
            ax.set_title(title, pad=6, fontsize=7)
            if i == 3:
                ax.set_xlabel('time (s)', labelpad=4, fontsize=7)

        plt.show()

    data = [x_vals.reshape(-1, 1), v_vals.reshape(-1, 1), a_vals.reshape(-1, 1), j_vals.reshape(-1, 1)]
    titles = ["position", "velocity", "acceleration", "jerk"]
    y_labels = ["position", "velocity", "acceleration", "jerk"]
    extra_data = [x_samples, v_samples, a_samples]
    extra_t = t_samples

    plot_subplots(t_vals, data, titles, y_labels, extra_data, extra_t)



# 设置numpy输出格式
np.set_printoptions(precision=4, suppress=True)

if __name__ == '__main__':
    dt = 50 
    N = 10
    delay_count = 3
    DOF = 7
    # 测试用例2 ：读取文件数据，进行轨迹规划和画图
    file_path = "trajData_wk-1-1-5.txt"
    positions, velocities = read_trajectory_data(file_path)
    if len(positions) == 0:
        print("错误：未读取到有效数据")
        exit(1)
    print(f"成功读取 {len(positions)} 个数据点")
    # 分析数据
    traj_list = []
    velocity_list = []
    last_p = positions[0, :]
    last_v = 0.5 * (positions[1, :] - last_p) / dt
    print(f"positions.shape = {positions.shape}, positions[i, :] = {positions[0, :]}")

    t_samples = np.arange(len(positions)) * dt / (1000)

    test_ruckig(t_samples, positions[:, 6], dt)
    




        
