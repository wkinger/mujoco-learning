from ruckig import Ruckig, InputParameter, OutputParameter, Result
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

def test_ruckig():
    # Same parameters as original test
    x0 = -1.0
    v0 = -1.0
    v_max = 3.0
    a_max = 20.0
    j_max = 330.0
    fps = 300.0
    dt = 1.0 / fps  # time step

    # Generate same input samples
    t_samples = np.arange(0, 4, 1 / 60)
    freq = 0.5
    amp = 0.5
    x_samples = np.sin(2 * np.pi * freq * t_samples) * amp
    v_samples = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t_samples) * amp
    a_samples = -(2 * np.pi * freq) ** 2 * np.sin(2 * np.pi * freq * t_samples) * amp

    # Initialize Ruckig with 1 DOF (degree of freedom)
    otg = Ruckig(1, dt)
    inp = InputParameter(1)
    out = OutputParameter(1)

    # Set limits
    inp.max_velocity = [v_max]
    inp.max_acceleration = [a_max]
    inp.max_jerk = [j_max]

    # Set initial state
    inp.current_position = [x0]
    inp.current_velocity = [v0]
    inp.current_acceleration = [0.0]

    # Storage for results
    res = []
    t_vals = []
    current_t = 0.0
    last_sample_t = 0.0
    prev_acceleration = 0.0  # Initialize previous acceleration for jerk calculation

    for i, (x_target, v_target, a_target, t_sample) in enumerate(zip(x_samples, v_samples, a_samples, t_samples)):
        # Set target for this waypoint
        inp.target_position = [float(x_target)]
        inp.target_velocity = [float(v_target)]
        inp.target_acceleration = [float(a_target)]

        # Calculate how many steps until next sample
        steps_needed = int((t_sample - last_sample_t) * fps)
        
        # Generate trajectory points
        for step in range(steps_needed):
            result = otg.update(inp, out)
            
            # Calculate jerk manually (rate of change of acceleration)
            current_acceleration = float(out.new_acceleration[0])
            jerk = (current_acceleration - prev_acceleration) / dt
            
            # Store results: [position, velocity, acceleration, jerk]
            res.append([
                out.new_position[0],
                out.new_velocity[0],
                current_acceleration,
                jerk
            ])
            t_vals.append(current_t)
            
            # Update input for next iteration
            inp.current_position = out.new_position
            inp.current_velocity = out.new_velocity
            inp.current_acceleration = out.new_acceleration
            prev_acceleration = current_acceleration  # Update for next jerk calculation
            
            current_t += dt
        
        last_sample_t = t_sample

    res = np.array(res)
    x_vals = res[:, 0:1]
    v_vals = res[:, 1:2]
    a_vals = res[:, 2:3]
    j_vals = res[:, 3:4]
    t_vals = np.array(t_vals)

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
            
            ax.set_title(title, pad=6, fontsize=7)
            if i == 3:
                ax.set_xlabel('time (s)', labelpad=4, fontsize=7)

        plt.show()

    data = [x_vals, v_vals, a_vals, j_vals]
    titles = ["position", "velocity", "acceleration", "jerk"]
    y_labels = ["position", "velocity", "acceleration", "jerk"]
    extra_data = [x_samples, v_samples, a_samples]
    extra_t = t_samples

    plot_subplots(t_vals, data, titles, y_labels, extra_data, extra_t)

if __name__ == "__main__":
    test_ruckig()
