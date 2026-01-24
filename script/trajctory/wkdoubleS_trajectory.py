import math
import matplotlib.pyplot as plt
import numpy as np
import copy

epsilon = 1e-10  # 极小值容差


class DoubleSCurveTrajectoryGenerator:
    def __init__(self, dof, q0_in, q1_in, v0_in, v1_in, vmax, vmin, amax, amin, jmax, jmin):
        self.dof = dof
        self.traj = []
        self.lambda_ = 0.99
        self.max_iter = 1000
        self.still = False
        self.reset(q0_in, q1_in, v0_in, v1_in, vmax, vmin, amax, amin, jmax, jmin)
    
    def reset(self, q0_in, q1_in, v0_in, v1_in, vmax, vmin, amax, amin, jmax, jmin):
        # 1. 给定初始条件，做相应转换
        self.sigma = np.sign(q1_in - q0_in)
        if abs(self.sigma) < epsilon:
            self.sigma = 1
        self.q0 = q0_in * self.sigma
        self.q1 = q1_in * self.sigma
        self.v0 = v0_in * self.sigma
        self.v1 = v1_in * self.sigma
        self.v_max = 0.5 * (self.sigma + 1) * vmax + 0.5 * (self.sigma - 1) * vmin
        self.v_min = 0.5 * (self.sigma + 1) * vmin + 0.5 * (self.sigma - 1) * vmax
        self.a_max = 0.5 * (self.sigma + 1) * amax + 0.5 * (self.sigma - 1) * amin
        self.a_min = 0.5 * (self.sigma + 1) * amin + 0.5 * (self.sigma - 1) * amax
        self.j_max = 0.5 * (self.sigma + 1) * jmax + 0.5 * (self.sigma - 1) * jmin
        self.j_min = 0.5 * (self.sigma + 1) * jmin + 0.5 * (self.sigma - 1) * jmax

        self.T = 0.
        self.T_a = 0.
        self.T_d = 0.
        self.T_j1 = 0.
        self.T_j2 = 0.
        self.T_v = 0.
        self.T_j = 0.

        self.v_lim = 0.
        self.a_lim_a = 0.
        self.a_lim_d = 0.
        self.still = False
    
    def is_valid(self):
        T_jstar = np.zeros(self.dof)
        T_jstar = np.min([np.sqrt(abs(self.v1 - self.v0) / self.j_max), self.a_max/self.j_max])
        if T_jstar < self.a_max/self.j_max:
            if abs(self.q1 - self.q0) < abs(T_jstar * (self.v1 + self.v0)) + epsilon:
                return False
        else:
            if abs(self.q1 - self.q0) < abs((T_jstar + (self.v1 - self.v0)/self.a_max) * (self.v1 + self.v0) / 2) + epsilon:
                return False
        return True

    def double_s_curve_trajectory(self):
        if abs(self.q0 - self.q1) < epsilon:
            print(f"q0 and q1 are the same")
            self.still = True
            return self.T
        # 2. 假设v_max与a_max可达，计算各段时间值
        if (self.v_max - self.v0) * self.j_max < self.a_max ** 2:
            print(f"无法达到a_max")
            self.T_j1 = np.sqrt((self.v_max - self.v0) / self.j_max)
            self.T_a = 2 * self.T_j1
        else:
            self.T_j1 = self.a_max / self.j_max
            self.T_a = self.T_j1 + (self.v_max - self.v0) / self.a_max

        if (self.v_max - self.v1) * self.j_max < self.a_max ** 2:
            print(f"无法达到a_min")
            self.T_j2 = np.sqrt((self.v_max - self.v1) / self.j_max)
            self.T_d = 2 * self.T_j2
        else:
            self.T_j2 = self.a_max / self.j_max
            self.T_d = self.T_j2 + (self.v_max - self.v1) / self.a_max
        
        self.T_v = (self.q1 - self.q0)/self.v_max - 0.5 * self.T_a * (1 + self.v0/self.v_max) - 0.5 * self.T_d * (1 + self.v1/self.v_max)

        # 3. 分情况处理
        if self.T_v > 0:
            print(f"可以达到v_max")
            self.T = self.T_a + self.T_d + self.T_v
            print(f"Ta = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}, Tj1 = {self.T_j1:.4f}, Tj2 = {self.T_j2:.4f}")
            self.calc_max_vel_and_acc()
            print(f"calculation finished, total time is {self.T}")
            return self.T
        
        self.T_v = 0.
        print(f"无法达到v_max")
        self.T = self.T_a + self.T_d + self.T_v
        print(f"T = {self.T:.4f} Ta = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}, Tj1 = {self.T_j1:.4f}, Tj2 = {self.T_j2:.4f}")
        iteration = 0
        while True:
            if iteration >= self.max_iter:
                break
            self.calc_traj_para()
            # self.print_info()
            if self.T_a < 0 or self.T_d < 0:
                if self.T_a < 0 and self.v0 > self.v1:
                    print("不存在加速段")
                    self.T_a = 0.
                    self.T_d = 2 * (self.q1 - self.q0)/(self.v1 + self.v0)
                    self.T_j1 = 0.
                    value = self.j_max * (self.j_max * ((self.q1 - self.q0)**2) + ((self.v1 + self.v0)**2) * (self.v1 - self.v0))
                    if value < 0.:
                        print(f"3.28b根号内为负，无法计算轨迹")
                        self.T = 0
                        break
                    self.T_j2 = (self.j_max * (self.q1 - self.q0) - np.sqrt(value))/(self.j_max * (self.v1 + self.v0))
                if self.T_d < 0 and self.v1 > self.v0:
                    print("不存在减速段")
                    self.T_d = 0.
                    self.T_a = 2 * (self.q1 - self.q0)/(self.v1 + self.v0)
                    self.T_j2 = 0.
                    value = self.j_max * (self.j_max * (self.q1 - self.q0)**2 - (self.v1 + self.v0)**2 * (self.v1 - self.v0))
                    if value < 0.:
                        print(f"3.29b根号内为负，无法计算轨迹")
                        self.T = 0
                        break
                    self.T_j1 = (self.j_max * (self.q1 - self.q0) - np.sqrt(value))/(self.j_max * (self.v1 + self.v0))
                self.calc_max_vel_and_acc()
                print(f"calculation finished, total time is {self.T}")
                return self.T
            else:
                if self.T_a > 2 * self.T_j and self.T_d > 2 * self.T_j:
                    print(f"存在加减速段")
                    self.calc_max_vel_and_acc()
                    print(f"calculation finished, total time is {self.T}")
                    break
                else:
                    iteration += 1
                    # print(f"iteration {iteration} start")
                    self.a_max *= self.lambda_
                    self.a_min *= self.lambda_
        return self.T
            
    def calc_traj_para(self):
        self.T_j1 = self.T_j2 = self.T_j = self.a_max / self.j_max
        delta = (self.a_max**4)/(self.j_max**2) + 2 * (self.v0**2 + self.v1**2) + \
        self.a_max * (4 * (self.q1 - self.q0) - (2 * self.a_max * (self.v0 + self.v1))/self.j_max)
        if delta < 0.:
            print(f"3.27根号内为负，无法计算轨迹")
            self.T = 0
            return self.T
        self.T_a = ((self.a_max**2)/self.j_max - 2 * self.v0 + np.sqrt(delta))/(2 * self.a_max)
        self.T_d = ((self.a_max**2)/self.j_max - 2 * self.v1 + np.sqrt(delta))/(2 * self.a_max)
        # print(f"calc_traj_para T_a = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}, Tj1 = {self.T_j1:.4f}, Tj2 = {self.T_j2:.4f}")
        self.T = self.T_a + self.T_d + self.T_v

    def calc_max_vel_and_acc(self):
        self.T = self.T_a + self.T_d + self.T_v
        self.a_lim_a = self.j_max * self.T_j1
        self.a_lim_d = -self.j_max * self.T_j2
        self.v_lim = self.v0 + (self.T_a - self.T_j1) * self.a_lim_a

    def get_profile(self, dt, T = None):
        if self.still:
            self.T = T
        factor = 1.
        if T != None:
            factor = self.T / T
            if factor > 1. or factor <= 0.:
                print(f"factor must be in (0,1]")
                return None
        t = 0
        t_list = []
        q_list = []
        dq_list = []
        ddq_list = []
        dddq_list = []
        while t < self.T - epsilon:
            if self.still:
                q_t = self.q0
                dq_t = ddq_t = dddq_t = 0.
            else:
                q_t, dq_t, ddq_t, dddq_t = self.get_traj_by_time_scale(t, factor)
            t_list.append(t / factor)
            q_list.append(q_t)
            dq_list.append(dq_t)
            ddq_list.append(ddq_t)
            dddq_list.append(dddq_t)
            t += dt
        if self.still:
            q_t = self.q0
            dq_t = ddq_t = dddq_t = 0.
        else:
            q_t, dq_t, ddq_t, dddq_t = self.get_traj_by_time_scale(self.T, factor)
        t_list.append(self.T / factor)
        q_list.append(q_t)
        dq_list.append(dq_t)
        ddq_list.append(ddq_t)
        dddq_list.append(dddq_t)
        return t_list, q_list, dq_list, ddq_list, dddq_list
    
    def get_traj_by_time_scale(self, t, factor):
        if t >= 0 and t <= self.T_j1:
            q_t = self.q0 + self.v0 * t + (self.j_max * t**3)/6.
            dq_t = self.v0 + (self.j_max * t**2)/2.
            ddq_t = self.j_max * t
            dddq_t = self.j_max
            # print(f"加加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v0 {self.v0:.4f}, a_lim_a = {self.a_lim_a:.4f}, T_j1 = {self.T_j1:.4f}")
        if t > self.T_j1 and t <= self.T_a - self.T_j1:
            q_t = self.q0 + self.v0 * t + self.a_lim_a * (3 * t**2 - 3 * self.T_j1 * t + self.T_j1**2)/6.
            dq_t = self.v0 + self.a_lim_a * (t - self.T_j1/2.)
            ddq_t = self.a_lim_a
            dddq_t = 0
            # print(f"恒加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v0 {self.v0:.4f}, a_lim_a = {self.a_lim_a:.4f}, T_j1 = {self.T_j1:.4f}")
        if t > self.T_a - self.T_j1 and t <= self.T_a:
            q_t = self.q0 + 0.5 * (self.v_lim + self.v0) * self.T_a - self.v_lim * (self.T_a - t) - (self.j_min * (self.T_a - t)**3)/6.
            dq_t = self.v_lim + (self.j_min * (self.T_a - t)**2)/2.
            ddq_t = -self.j_min * (self.T_a - t)
            dddq_t = self.j_min
            # print(f"减加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f}  v_lim {self.v_lim:.4f}, j_min = {self.j_min:.4f}, T_a = {self.T_a:.4f}")
        if t > self.T_a and t <= self.T_a + self.T_v:
            q_t = self.q0 + 0.5 * (self.v_lim + self.v0) * self.T_a + self.v_lim * (t - self.T_a) 
            dq_t = self.v_lim 
            ddq_t = 0
            dddq_t = 0
            # print(f"恒速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v_lim {self.v_lim:.4f}, j_min = {self.j_min:.4f}, T_a = {self.T_a:.4f}")
        if t > self.T - self.T_d and t <= self.T - self.T_d + self.T_j2:
            q_t = self.q1 - 0.5 * (self.v_lim + self.v1) * self.T_d + self.v_lim * (t - self.T + self.T_d) - (self.j_max * (t - self.T + self.T_d)**3)/6.
            dq_t = self.v_lim - (self.j_max * (t - self.T + self.T_d)**2)/2      
            ddq_t = -self.j_max * (t - self.T + self.T_d)
            dddq_t = self.j_min
            # print(f"加减速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v_lim {self.v_lim:.4f}, a_lim_d = {self.a_lim_d:.4f}, j_max = {self.j_max:.4f}")
        if t > self.T - self.T_d + self.T_j2 and t <= self.T - self.T_j2:
            q_t = self.q1 - 0.5 * (self.v_lim + self.v1) * self.T_d + self.v_lim * (t - self.T + self.T_d) + \
            (self.a_lim_d/6.) * (3 * ((t - self.T + self.T_d)**2) - 3 * self.T_j2 * (t - self.T + self.T_d) + self.T_j2**2)
            dq_t = self.v_lim + self.a_lim_d * (t - self.T + self.T_d - self.T_j2/2)
            ddq_t = self.a_lim_d
            dddq_t = 0
            # print(f"恒减速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f}  q1 {self.q1:.4f}, v1 {self.v1:.4f}, v_lim {self.v_lim:.4f}")
        if t > self.T - self.T_j2 and t <= self.T:
            q_t = self.q1 - self.v1 * (self.T - t) - (self.j_max * (self.T - t)**3)/6.
            dq_t = self.v1 + (self.j_max * (self.T - t)**2)/2.
            ddq_t = -self.j_max * (self.T - t)
            dddq_t = self.j_max
            # print(f"减减速t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v_lim {self.v_lim:.4f}, j_max = {self.j_max:.4f}, T_d = {self.T_d:.4f}")

        q_t = self.sigma * q_t 
        dq_t = self.sigma * dq_t * factor
        ddq_t = self.sigma * ddq_t * factor**2
        dddq_t = self.sigma * dddq_t * factor**3
        return q_t, dq_t, ddq_t, dddq_t
    

    
    def print_info(self):
        print(f"print trajectory info:")
        print(f"q0 = {self.q0:.4f}, q1 = {self.q1:.4f}, v0 = {self.v0:.4f}, v1 = {self.v1:.4f}")
        print(f"T = {self.T:.4f}, Ta = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}, \
        T_j = {self.T_j:.4f} Tj1 = {self.T_j1:.4f}, Tj2 = {self.T_j2:.4f}")
        print(f"vlim = {self.v_lim:.4f}, a_lim_a = {self.a_lim_a:.4f}, a_lim_d = {self.a_lim_d:.4f}")
        print(f"vmax = {self.v_max:.4f}, vmin = {self.v_min:.4f}, a_max = {self.a_max:.4f}, a_min = {self.a_min:.4f}, jmax = {self.j_max:.4f}, jmin = {self.j_min:.4f}")
        print(f"加加速 {self.T_j1:.4f} 恒加速 {self.T_a - self.T_j1:.4f} 减加速 {self.T_a:.4f} 恒速 {self.T_a + self.T_v:.4f} \
    {self.T - self.T_d} 加减速 {self.T - self.T_d + self.T_j2:.4f} 恒减速 {self.T - self.T_j2:.4f} 减减速 {self.T:.4f}")

    def plot_all_trajectories(self, t_list, pos, vel, acc, jerk, t_list1=[], pos1=[], vel1=[], acc1=[], jerk1=[]):
        # 7. 绘图
        plt.figure(figsize=(12, 8))
        # 位置
        plt.subplot(4, 1, 1)
        plt.plot(t_list, pos, 'b-', linewidth=1.5, label=f'Position')
        if len(t_list1) > 0:
            plt.plot(t_list1, pos1, 'b--', linewidth=1.5, label=f'Position (scale))')
        plt.axhline(pos[0], color='b', linestyle='--', alpha=0.5, label='Start Pos')
        plt.axhline(pos[-1], color='r', linestyle='--', alpha=0.5, label='Target Pos')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [a.u.]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 速度
        plt.subplot(4, 1, 2)
        plt.plot(t_list, vel, 'r-', linewidth=1.5, label='Velocity')
        if len(t_list1) > 0:
            plt.plot(t_list1, vel1, 'r--', linewidth=1.5, label=f'Velocity (scale))')
        # plt.axhline(self.v_lim, color='orange', linestyle='--', alpha=0.5, label=f'Max Vel')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [a.u./s]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 加速度
        plt.subplot(4, 1, 3)
        plt.plot(t_list, acc, 'g-', linewidth=1.5, label='Acceleration')
        if len(t_list1) > 0:
            plt.plot(t_list1, acc1, 'g--', linewidth=1.5, label=f'Acceleration (scale))')
        # plt.axhline(self.a_max, color='orange', linestyle='--', alpha=0.5, label=f'Max Acc')
        # plt.axhline(self.a_min, color='purple', linestyle='--', alpha=0.5, label=f'Min Acc')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [a.u./s²]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 加加速度
        plt.subplot(4, 1, 4)
        plt.plot(t_list, jerk, 'm-', linewidth=1.5, label='Jerk')
        if len(t_list1) > 0:
            plt.plot(t_list1, jerk1, 'm--', linewidth=1.5, label=f'Jerk (scale))')
        # plt.axhline(self.j_max, color='orange', linestyle='--', alpha=0.5, label=f'Max Jerk')
        # plt.axhline(self.j_min, color='purple', linestyle='--', alpha=0.5, label=f'Min Jerk')
        plt.xlabel('Time [s]')
        plt.ylabel('Jerk [a.u./s³]')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.title(f'Double S-Curve Trajectory')
        plt.show()

        return {
            'time': t_list,
            'position': pos,
            'velocity': vel,
            'acceleration': acc,
            'jerk': jerk,
            'total_time': t_list,
        }

    def calculate_differential_velocity(self, t_list, pos_list): 
        """ 计算差分速度（数值微分速度）
        参数:
            t_list: 时间序列
            pos_list: 位置序列
            
        返回:
            diff_vel: 差分速度序列
        """
        if len(t_list) < 2:
            return []
        
        diff_vel = []
        n = len(t_list)
        
        # 使用中心差分法计算速度（更精确）
        for i in range(n):
            if i == 0:
                # 前向差分
                dt = t_list[1] - t_list[0]
                if dt > 0:
                    diff_vel.append((pos_list[1] - pos_list[0]) / dt)
                else:
                    diff_vel.append(0.0)
            elif i == n - 1:
                # 后向差分
                dt = t_list[-1] - t_list[-2]
                if dt > 0:
                    diff_vel.append((pos_list[-1] - pos_list[-2]) / dt)
                else:
                    diff_vel.append(0.0)
            else:
                # 中心差分（更精确）
                dt_forward = t_list[i] - t_list[i-1]
                dt_backward = t_list[i+1] - t_list[i]
                if dt_forward > 0 and dt_backward > 0:
                    # 加权平均
                    weight_forward = dt_backward / (dt_forward + dt_backward)
                    weight_backward = dt_forward / (dt_forward + dt_backward)
                    vel_forward = (pos_list[i] - pos_list[i-1]) / dt_forward
                    vel_backward = (pos_list[i+1] - pos_list[i]) / dt_backward
                    diff_vel.append(weight_forward * vel_forward + weight_backward * vel_backward)
                else:
                    diff_vel.append(0.0)
        return diff_vel
    def compare_velocities(self, t_list, pos_list, theoretical_vel_list):
        """
        比较理论速度和差分速度
        
        参数:
            t_list: 时间序列
            pos_list: 位置序列
            theoretical_vel_list: 理论速度序列
            
        返回:
            comparison_dict: 包含比较结果的字典
        """
        # 计算差分速度
        diff_vel = self.calculate_differential_velocity(t_list, pos_list)
        
        if len(diff_vel) != len(theoretical_vel_list):
            print("警告：差分速度和理论速度序列长度不一致")
            min_len = min(len(diff_vel), len(theoretical_vel_list))
            diff_vel = diff_vel[:min_len]
            theoretical_vel_list = theoretical_vel_list[:min_len]
            t_list = t_list[:min_len]
        
        # 计算误差
        errors = []
        relative_errors = []
        for i in range(len(diff_vel)):
            error = abs(diff_vel[i] - theoretical_vel_list[i])
            errors.append(error)
            if abs(theoretical_vel_list[i]) > 1e-10:  # 避免除以0
                relative_error = error / abs(theoretical_vel_list[i]) * 100
            else:
                relative_error = 0.0
            relative_errors.append(relative_error)
        
        # 统计信息
        max_error = max(errors) if errors else 0.0
        avg_error = sum(errors) / len(errors) if errors else 0.0
        max_relative_error = max(relative_errors) if relative_errors else 0.0
        avg_relative_error = sum(relative_errors) / len(relative_errors) if relative_errors else 0.0
        
        return {
            'time': t_list,
            'theoretical_velocity': theoretical_vel_list,
            'differential_velocity': diff_vel,
            'absolute_errors': errors,
            'relative_errors_percent': relative_errors,
            'max_absolute_error': max_error,
            'average_absolute_error': avg_error,
            'max_relative_error_percent': max_relative_error,
            'average_relative_error_percent': avg_relative_error
        }

    def plot_velocity_comparison(self, t_list, pos_list, theoretical_vel_list):
        """
        绘制理论速度和差分速度的比较图
        """
        comparison = self.compare_velocities(t_list, pos_list, theoretical_vel_list)
        
        plt.figure(figsize=(12, 10))
        
        # 速度比较
        plt.subplot(3, 1, 1)
        plt.plot(comparison['time'], comparison['theoretical_velocity'], 'b-', 
                linewidth=2, label='Theoretical Velocity')
        plt.plot(comparison['time'], comparison['differential_velocity'], 'r--', 
                linewidth=1.5, label='Differential Velocity')
        plt.xlabel('time [s]')
        plt.ylabel('vel [a.u./s]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Theoretical Velocity vs Differential Velocity')
        
        # 绝对误差
        plt.subplot(3, 1, 2)
        plt.plot(comparison['time'], comparison['absolute_errors'], 'g-', 
                linewidth=1.5, label='Absolute Error')
        plt.axhline(comparison['average_absolute_error'], color='orange', 
                linestyle='--', alpha=0.7, label=f'Average Error: {comparison["average_absolute_error"]:.6f}')
        plt.xlabel('time [s]')
        plt.ylabel('Absolute Error [a.u./s]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Speed Absolute Error')
        
        # 相对误差
        plt.subplot(3, 1, 3)
        plt.plot(comparison['time'], comparison['relative_errors_percent'], 'm-', 
                linewidth=1.5, label='Relative Error [%]')
        plt.axhline(comparison['average_relative_error_percent'], color='red', 
                linestyle='--', alpha=0.7, label=f'average vel: {comparison["average_relative_error_percent"]:.2f}%')
        plt.xlabel('time [s]')
        plt.ylabel('Relative Error [%]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Speed Relative Error')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"速度比较统计:")
        print(f"最大绝对误差: {comparison['max_absolute_error']:.6f}")
        print(f"平均绝对误差: {comparison['average_absolute_error']:.6f}")
        print(f"最大相对误差: {comparison['max_relative_error_percent']:.2f}%")
        print(f"平均相对误差: {comparison['average_relative_error_percent']:.2f}%")
        
        return comparison

# ======================== 测试用例（反向运动重点验证） ========================
if __name__ == "__main__":
    # 输入v0_in=2 → 校准后v0=-2（因为q1<q0，dir_pos=-1），位置从15→5
    constrain = [0,0.,0,0,10,-10,10,-10,30,-30]  # 有恒速段 正向
    # constrain = [10,0,-1,0,5,-5,10,-10,30,-30]  # 有恒速段 反向
    # constrain = [0,10,-3,0,5,-5,10,-10,30,-30]  # 有恒速段 正向 反速度 
    # constrain = [10,0,3,0,5,-5,10,-10,30,-30]  # 有恒速段 反向 反速度 

    # constrain = [0,10,1,0,10,-10,10,-10,30,-30] # 无恒速段 正向
    # constrain = [10,0,-1,0,10,-10,10,-10,30,-30] # 无恒速段 反向  
    # constrain = [0,10,-2,0,10,-10,10,-10,30,-30] # 无恒速段 正向 反速度
    # constrain = [10,0,2,0,10,-10,10,-10,30,-30] # 无恒速段 反向 反速度

    # constrain = [0,10,7,0,10,-10,10,-10,30,-30] # 无恒加速段 正向
    # constrain = [10,0,-7,0,10,-10,10,-10,30,-30] # 无恒加速段 反向
    # constrain = [0,10,-2,0,10,-10,20,-20,30,-30] # 无恒加速段 正向 反速度
    # constrain = [10,0,2,0,10,-10,20,-20,30,-30] # 无恒加速段 反向 反速度

    # constrain = [0,10,7.5,0,10,-10,10,-10,30,-30] # 仅有减速段
    # constrain = [0,10,0,7.5,10,-10,10,-10,30,-30] # 仅有加速段
    # constrain = [10,0,0,-7.5,10,-10,10,-10,30,-30] # 仅有减速段 反向
    # constrain = [10,0,-7.5,0,10,-10,10,-10,30,-30] # 仅有加速段 正向
    # constrain = [10,0,-7.5,0,10,-10,10,-10,30,-30] # 仅有加速段 正向
    # constrain = [0,10,3,10,10,-10,10,-10,30,-30] # 仅有加速段 正向 反速度

    # constrain = [0,5.77351,0,10,10,-10,10,-10,30,-30] # 非常极限的情况
    # constrain = [0,5.77351,0,10,10,-10,10,-10,30,-30] # 非常极限的情况
    constrain = [0, 10, 0.0, 0.0, 5,-5,10,-10,30,-30]  # 有恒速段 正向

    planner = DoubleSCurveTrajectoryGenerator(7, constrain[0], constrain[1], constrain[2], constrain[3], \
                    constrain[4], constrain[5], constrain[6], constrain[7], constrain[8], constrain[9])
    total_time = planner.double_s_curve_trajectory()
    planner.print_info()
    is_valid = planner.is_valid()
    if is_valid:
        print("Trajectory is valid.")
    else:
        print("Trajectory is not valid.")
    t_list, pos, vel, acc, jerk = planner.get_profile(0.001)
    print(f"pos {len(pos)}")
        # 计算并比较速度
    comparison_result = planner.compare_velocities(t_list, pos, vel)

    # 或者直接绘制比较图
    planner.plot_velocity_comparison(t_list, pos, vel)
    # t_list1, pos1, vel1, acc1, jerk1 = planner.get_profile(0.01,3)
    # for i in range(len(t_list)):
    #     print(f"t = {t_list[i]:.4f}, q = {pos[i]:.4f}, v = {vel[i]:.4f}, a = {acc[i]:.4f}, j = {jerk[i]:.4f}")
    # planner.plot_all_trajectories(t_list, pos, vel, acc, jerk)


