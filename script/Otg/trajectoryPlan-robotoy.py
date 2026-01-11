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
    

from robotoy.itp_state.itp_state import ItpState
import numpy as np

from matplotlib import pyplot as plt
from loguru import logger
import matplotlib.font_manager as fm
from matplotlib import rcParams

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

Flt = np.float32
EPS = 1e-6

def arr(*l):
    return np.array(l, dtype=Flt)

def test_itp_state(t_samples, x_samples, dt):
    x0 = arr(x_samples[0])
    v0 = arr(0)
    v_max = arr(3.0)
    a_max = arr(20.0)
    j_max = arr(330)
    fps = 1000.0 / dt
    itp_state = ItpState()
    itp_state.init(v_max=v_max, a_max=a_max, j_max=j_max, fps=fps)
    itp_state.init(x0=x0, v0=v0)

    a_samples = np.zeros_like(x_samples)
    v_samples = np.zeros_like(t_samples)

    last_itpltn_t = 0
    res = []
    t_vals = []
    for i, (x, v, a, t) in enumerate(zip(x_samples, v_samples, a_samples, t_samples)):
        points_needed = int((t - last_itpltn_t) * fps)
        # logger.info(f"start: {last_itpltn_t}")
        if i < len(x_samples) - 1:
            v_target = arr(x_samples[i + 1] - x_samples[i - 1]/(dt*2))
        else:
            v_target = arr(0)
        v = v_target
        v_samples[i] = v_target

        res += itp_state.interpolate(
            arr(x), arr(v), arr(a), points_needed, first_delta_t=(last_itpltn_t - t)
        )
        t_vals += [last_itpltn_t + i / fps for i in range(points_needed)]
        last_itpltn_t += points_needed / fps

    res = np.array(res)
    x_vals = np.array(res[:, 0])
    v_vals = np.array(res[:, 1])
    a_vals = np.array(res[:, 2])
    j_vals = np.array(res[:, 3])

    # [DEBUG] for plot
#     fixed_j_vals = np.zeros_like(j_vals)
#     for i in range(0, len(j_vals) - 1):
#         if (i > 0 and j_vals[i - 1] * j_vals[i] < 0) or (i > 1 and j_vals[i - 1] * j_vals[i - 2] < 0) or (i > 2 and j_vals[i - 2] * j_vals[i - 3] < 0):
#             fixed_j_vals[i] = 0
#         else:
#             fixed_j_vals[i] = j_vals[i]
#     j_vals = fixed_j_vals
# 
    def plot_subplots(t, data, titles, y_labels, extra_data=None, extra_t=None):
        # 设置字体和样式
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 8,                 
            'font.family': 'serif',
            'mathtext.fontset': 'stix',
            'axes.labelsize': 9,           
            'axes.titlesize': 10,          
            'xtick.labelsize': 8,          
            'ytick.labelsize': 8,
            'legend.fontsize': 7,          
            # 'figure.figsize': (6, 8),      
            'figure.dpi': 300,
            'lines.linewidth': 1.0,        
            'lines.markersize': 3,         
            'axes.linewidth': 0.8,         
            'grid.linewidth': 0.4,         
        })

        fig, axs = plt.subplots(4, 1, figsize=(5, 6))
        fig.subplots_adjust(hspace=0.60)       # 稍微减小子图间距

        # ...existing code...

        for i, (ax, data_vals, title, y_label) in enumerate(zip(axs, data, titles, y_labels)):
            for j in range(data_vals.shape[1]):
                ax.plot(
                    t,
                    data_vals[:, j],
                    color='black',
                    marker='.',
                    markersize=2,              # 减小数据点大小
                    linestyle='-',
                    zorder=3,
                )
            
            if extra_data is not None and extra_t is not None and i < len(extra_data):
                ax.plot(
                    extra_t,
                    extra_data[i],
                    'x',
                    color='#FF2C00',
                    markersize=3,              # 减小采样点大小
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
            
            ax.set_title(title, pad=6, fontsize=7)
            if i == 3:
                ax.set_xlabel('time (s)', labelpad=4, fontsize=7)
                time_segments = [
                    (0.00, 0.06, 't_7'),
                    (0.06, 0.14, 't_6'),
                    (0.14, 0.20, 't_5'),
                    (0.20, 0.80, 't_4'),
                    (0.80, 0.86, 't_3'),
                    (0.86, 0.94, 't_2'),
                    (0.94, 0.99, 't_1')
                ]
                
                # [t label]
                # ymin = ax.get_ylim()[0]
                # y_pos = ymin - (ax.get_ylim()[1] - ymin) * 0.20  # 调整标注位置
                
                # for t_start, t_end, label in time_segments:
                #     # 添加箭头标注
                #     ax.annotate('', 
                #         xy=(t_start, y_pos), 
                #         xytext=(t_end, y_pos),
                #         arrowprops=dict(arrowstyle='<->',
                #                       linewidth=0.8,
                #                       shrinkA=0,
                #                       shrinkB=0))
                    
                #     # 添加文本标签
                #     mid_point = (t_start + t_end) / 2
                #     ax.text(mid_point, y_pos * 0.62,
                #            r"$%s$" % label,
                #            horizontalalignment='center',
                #            verticalalignment='top',
                #            fontsize=8)
                
                #
                # ax.set_ylim(bottom=y_pos * 1.2)
              
            
            # 优化图例
            # ax.legend(frameon=True, 
            #          fancybox=False, 
            #          edgecolor='black',
            #          ncol=2, 
            #          loc='upper right', 
            #          bbox_to_anchor=(0.99, 0.99),
            #          fontsize=7,               # 明确设置图例字体大小
            #          borderaxespad=0.1)        # 减小图例边距

        plt.show()


    data = [x_vals, v_vals, a_vals, j_vals]
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

    test_itp_state(t_samples, positions[:, 6], dt)
    




        
