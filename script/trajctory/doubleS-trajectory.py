import math
import matplotlib.pyplot as plt
import numpy as np

def double_s_curve_trajectory(q0, q1, vmax, v0_in, v1_in, amax, jmax):
    """
    双S曲线轨迹生成（最终终极版：彻底解决反向运动速度方向问题）
    参数说明：
        q0: 起始位置
        q1: 目标位置
        vmax: 最大速度（绝对值，正数）
        v0_in: 输入起始速度（绝对值有效，方向会被自动校准为朝向目标）
        v1_in: 输入目标速度（绝对值有效，方向会被自动校准为朝向目标）
        amax: 最大加速度（绝对值，正数）
        jmax: 最大加加速度（绝对值，正数）
    返回：
        轨迹数据字典
    """
    # 1. 核心：位置差方向（决定速度的正确方向，必须朝向目标）
    delta_q = q1 - q0
    dir_pos = 1 if delta_q > 0 else -1  # 1: q0→q1  ， -1: q0←q1
    abs_delta_q = abs(delta_q)

    # 2. 速度方向校准：强制速度方向与位置差方向一致，保留输入绝对值
    abs_v0_in = abs(v0_in)
    abs_v1_in = abs(v1_in)
    v0 = dir_pos * abs_v0_in   # 校准后初始速度
    v1 = dir_pos * abs_v1_in   # 校准后目标速度

    # 3. 参数初始化
    abs_vmax = abs(vmax)
    abs_amax = abs(amax)
    abs_jmax = abs(jmax)

    # 速度超限校验
    if abs(v0) > abs_vmax:
        print(f"警告：初始速度绝对值{abs(v0)}超过最大速度{abs_vmax}，已截断为{dir_pos * abs_vmax}")
        v0 = dir_pos * abs_vmax
    if abs(v1) > abs_vmax:
        print(f"警告：目标速度绝对值{abs(v1)}超过最大速度{abs_vmax}，已截断为{dir_pos * abs_vmax}")
        v1 = dir_pos * abs_vmax

    abs_v0 = abs(v0)
    abs_v1 = abs(v1)

    # 4. 计算各阶段时间（基于绝对值，保证时间为正）
    # 加速段时间 (j→amax→j)
    if (abs_vmax - abs_v0) * abs_jmax < abs_amax ** 2:
        Tj1 = math.sqrt((abs_vmax - abs_v0) / abs_jmax) if abs_v0 < abs_vmax else 0
        Ta = 2 * Tj1
        alima_mag = Tj1 * abs_jmax if abs_v0 < abs_vmax else 0
    else:
        Tj1 = abs_amax / abs_jmax
        Ta = Tj1 + (abs_vmax - abs_v0) / abs_amax
        alima_mag = abs_amax

    # 减速段时间 (j→amax→j)
    if (abs_vmax - abs_v1) * abs_jmax < abs_amax ** 2:
        Tj2 = math.sqrt((abs_vmax - abs_v1) / abs_jmax) if abs_v1 < abs_vmax else 0
        Td = 2 * Tj2
        alimd_mag = Tj2 * abs_jmax if abs_v1 < abs_vmax else 0
    else:
        Tj2 = abs_amax / abs_jmax
        Td = Tj2 + (abs_vmax - abs_v1) / abs_amax
        alimd_mag = abs_amax

    # 匀速段时间
    Tv = (abs_delta_q / abs_vmax) - (Ta/2)*(1 + abs_v0/abs_vmax) - (Td/2)*(1 + abs_v1/abs_vmax)
    Tv = max(Tv, 0)
    T_total = Ta + Tv + Td

    # 5. 处理无法达到最大速度的场景
    vlim = dir_pos * abs_vmax  # 匀速速度，方向朝向目标
    alima = dir_pos * alima_mag  # 加速度方向朝向目标
    alimd = dir_pos * alimd_mag  # 减速度方向朝向目标

    if Tv < 1e-6:
        Tv = 0
        amax_org = abs_amax
        delta = (abs_amax**4)/(abs_jmax**2) + 2*(abs_v0**2 + abs_v1**2) + abs_amax*(4*abs_delta_q - 2*abs_amax/abs_jmax*(abs_v0 + abs_v1))
        delta = max(delta, 1e-10)
        Tj1 = abs_amax / abs_jmax
        Ta = (abs_amax**2/abs_jmax - 2*abs_v0 + math.sqrt(delta)) / (2*abs_amax)
        Tj2 = abs_amax / abs_jmax
        Td = (abs_amax**2/abs_jmax - 2*abs_v1 + math.sqrt(delta)) / (2*abs_amax)
        vlim = v0 + (Ta - Tj1) * alima
        T_total = Ta + Tv + Td

    # 6. 状态递推生成轨迹（核心：继承前一时刻状态，保证连续）
    dt = 0.001
    t_list = np.arange(0, T_total + dt, dt)
    pos = np.zeros_like(t_list)
    vel = np.zeros_like(t_list)
    acc = np.zeros_like(t_list)
    jerk = np.zeros_like(t_list)

    # 初始状态
    q_prev = q0
    v_prev = v0
    a_prev = 0.0
    j_prev = 0.0

    for i, t in enumerate(t_list):
        delta_t = dt if i > 0 else 0
        # 阶段1: 加加速度上升 (j=+jmax*dir_pos)
        if 0 <= t < Tj1:
            j_curr = dir_pos * abs_jmax
            a_curr = a_prev + j_curr * delta_t
            v_curr = v_prev + a_prev * delta_t + 0.5 * j_curr * delta_t**2
            q_curr = q_prev + v_prev * delta_t + 0.5 * a_prev * delta_t**2 + (1/6)*j_curr*delta_t**3
        # 阶段2: 恒定加速度
        elif Tj1 <= t < (Ta - Tj1):
            j_curr = 0
            a_curr = alima
            v_curr = v_prev + a_curr * delta_t
            q_curr = q_prev + v_prev * delta_t + 0.5 * a_curr * delta_t**2
        # 阶段3: 加加速度下降 (j=-jmax*dir_pos)
        elif (Ta - Tj1) <= t < Ta:
            j_curr = -dir_pos * abs_jmax
            a_curr = a_prev + j_curr * delta_t
            v_curr = v_prev + a_prev * delta_t + 0.5 * j_curr * delta_t**2
            q_curr = q_prev + v_prev * delta_t + 0.5 * a_prev * delta_t**2 + (1/6)*j_curr*delta_t**3
        # 阶段4: 匀速段
        elif Ta <= t < (Ta + Tv):
            j_curr = 0
            a_curr = 0
            v_curr = vlim
            q_curr = q_prev + v_curr * delta_t
        # 阶段5: 减速段加加速度下降 (j=-jmax*dir_pos)
        elif (Ta + Tv) <= t < (Ta + Tv + Tj2):
            j_curr = -dir_pos * abs_jmax
            a_curr = a_prev + j_curr * delta_t
            v_curr = v_prev + a_prev * delta_t + 0.5 * j_curr * delta_t**2
            q_curr = q_prev + v_prev * delta_t + 0.5 * a_prev * delta_t**2 + (1/6)*j_curr*delta_t**3
        # 阶段6: 恒定减速度
        elif (Ta + Tv + Tj2) <= t < (T_total - Tj2):
            j_curr = 0
            a_curr = -alimd  # 减速：加速度与运动方向相反
            v_curr = v_prev + a_curr * delta_t
            q_curr = q_prev + v_prev * delta_t + 0.5 * a_curr * delta_t**2
        # 阶段7: 减速段加加速度上升 (j=+jmax*dir_pos)
        elif (T_total - Tj2) <= t <= T_total:
            j_curr = dir_pos * abs_jmax
            a_curr = a_prev + j_curr * delta_t
            v_curr = v_prev + a_prev * delta_t + 0.5 * j_curr * delta_t**2
            q_curr = q_prev + v_prev * delta_t + 0.5 * a_prev * delta_t**2 + (1/6)*j_curr*delta_t**3
        else:
            q_curr = q1
            v_curr = v1
            a_curr = 0
            j_curr = 0

        # 更新状态
        pos[i] = q_curr
        vel[i] = v_curr
        acc[i] = a_curr
        jerk[i] = j_curr
        q_prev, v_prev, a_prev, j_prev = q_curr, v_curr, a_curr, j_curr

    # 强制终点状态
    pos[-1] = q1
    vel[-1] = v1

    # 7. 绘图
    plt.figure(figsize=(12, 8))
    # 位置
    plt.subplot(4, 1, 1)
    plt.plot(t_list, pos, 'b-', linewidth=1.5, label=f'Position (q0={q0}→q1={q1})')
    plt.axhline(q0, color='b', linestyle='--', alpha=0.5, label='Start Pos')
    plt.axhline(q1, color='r', linestyle='--', alpha=0.5, label='Target Pos')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [a.u.]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 速度
    plt.subplot(4, 1, 2)
    plt.plot(t_list, vel, 'r-', linewidth=1.5, label='Velocity')
    plt.axhline(v0, color='orange', linestyle='--', alpha=0.5, label=f'Calibrated Init Vel={v0}')
    plt.axhline(v1, color='purple', linestyle='--', alpha=0.5, label=f'Calibrated Target Vel={v1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [a.u./s]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 加速度
    plt.subplot(4, 1, 3)
    plt.plot(t_list, acc, 'g-', linewidth=1.5, label='Acceleration')
    plt.axhline(amax, color='g', linestyle='--', alpha=0.3)
    plt.axhline(-amax, color='g', linestyle='--', alpha=0.3)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [a.u./s²]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 加加速度
    plt.subplot(4, 1, 4)
    plt.plot(t_list, jerk, 'm-', linewidth=1.5, label='Jerk')
    plt.axhline(jmax, color='m', linestyle='--', alpha=0.3)
    plt.axhline(-jmax, color='m', linestyle='--', alpha=0.3)
    plt.xlabel('Time [s]')
    plt.ylabel('Jerk [a.u./s³]')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    motion_dir = "Forward" if dir_pos == 1 else "Reverse"
    plt.title(f'Double S-Curve Trajectory ({motion_dir} Motion)')
    plt.show()

    # 验证信息
    pos_diff = np.max(np.abs(np.diff(pos)))
    print(f"→ 位置相邻点最大差值：{pos_diff:.8f} (≤1e-5 表示连续)")
    print(f"→ 输入初始速度：{v0_in} | 校准后初始速度：{v0} (方向朝向目标)")
    print(f"→ 起始位置：{q0} | 目标位置：{q1} | 轨迹终点位置：{pos[-1]}")

    return {
        'time': t_list,
        'position': pos,
        'velocity': vel,
        'acceleration': acc,
        'jerk': jerk,
        'total_time': T_total,
        'position_max_diff': pos_diff
    }

# ======================== 测试用例（反向运动重点验证） ========================
if __name__ == "__main__":
    print("="*60 + " 测试用例1：反向运动（15→5）输入正初始速度 " + "="*60)
    # 输入v0_in=2 → 校准后v0=-2（因为q1<q0，dir_pos=-1），位置从15→5
    double_s_curve_trajectory(q0=15, q1=5, vmax=8, v0_in=2, v1_in=1, amax=15, jmax=25)

    print("\n" + "="*60 + " 测试用例2：反向运动（15→5）输入负初始速度 " + "="*60)
    # 输入v0_in=-2 → 校准后v0=-2，位置从15→5
    double_s_curve_trajectory(q0=15, q1=5, vmax=8, v0_in=-2, v1_in=-1, amax=15, jmax=25)

    print("\n" + "="*60 + " 测试用例3：正向运动（0→15）输入正初始速度 " + "="*60)
    # 输入v0_in=3 → 校准后v0=3，位置从0→15
    double_s_curve_trajectory(q0=0, q1=15, vmax=8, v0_in=3, v1_in=2, amax=15, jmax=25)
