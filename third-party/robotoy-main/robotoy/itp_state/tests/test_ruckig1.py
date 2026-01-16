import numpy as np
from matplotlib import pyplot as plt
from ruckig import Ruckig, InputParameter, OutputParameter, Result
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

Flt = np.float32
EPS = 1e-6

def arr(*l):
    return np.array(l, dtype=Flt)

def test_ruckig():
    # 设置初始条件 - 与 test_package.py 相同
    x0 = arr(-1)
    v0 = arr(-1)
    v_max = arr(3.0)
    a_max = arr(20.0)
    j_max = arr(330)
    fps = 20.0
    
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
    t_samples = np.arange(0, 4, 1 / 60)
    freq = 0.5
    amp = 0.5
    x_samples = np.sin(2 * np.pi * freq * t_samples) * amp

    v_samples = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t_samples) * amp
    # v_samples = np.zeros_like(x_samples)

    # a_samples = -4 * np.pi**2 * freq**2 * np.sin(2 * np.pi * freq * t_samples) * amp
    a_samples = np.zeros_like(x_samples)
    
    # 生成轨迹
    res = []
    t_vals = []
    out = OutputParameter(1)
    print(f"x_samples {x_samples.shape}")
    last_itpltn_t = 0
    for i, (x_target, v_target, a_target, t_target) in enumerate(zip(x_samples, v_samples, a_samples, t_samples)):
        points_needed = int((t_target - last_itpltn_t) * fps)
        
        # 设置目标状态
        inp.target_position = arr(x_target)
        inp.target_velocity = arr(v_target)
        inp.target_acceleration = arr(a_target)
        
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

    def plot_subplots(t, data, titles, y_labels, extra_data=None, extra_t=None):
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
                    marker='o',
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

if __name__ == "__main__":
    test_ruckig()