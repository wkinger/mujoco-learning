import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# 梯形速度轨迹有两种情况，
# 1、预设时长以及加速度  这里实现这种情况
# 2、预设最大加速度以及最大速度  则会计算出最小运动时间的轨迹
class TrapezoidalGenerator:
    def __init__(self, dof, q0_in, q1_in, v0_in, v1_in, amax, t0, t1):
        self.dof = dof
        self.traj = []
        self.reset(q0_in, q1_in, v0_in, v1_in, amax, t0, t1)
    
    def reset(self, q0_in, q1_in, v0_in, v1_in, amax, t0, t1):
        # 1. 给定初始条件，做相应转换
        self.sigma = np.sign(q1_in - q0_in)
        self.q0 = q0_in * self.sigma
        self.q1 = q1_in * self.sigma
        self.v0 = v0_in * self.sigma
        self.v1 = v1_in * self.sigma
        self.a_max = 0.5 * (self.sigma + 1) * amax - 0.5 * (self.sigma - 1) * amax
        self.h = self.q1 - self.q0
        self.t0 = t0
        self.t1 = t1
        self.T = self.t1 - self.t0
        if self.T <= 0.:
            print("input t0 and t1 is invalid")
        self.T_a = 0.
        self.T_d = 0.
        self.T_v = 0.

        self.v_v = 0.
        self.a_lim = 0.

    def trapezoidal_trajc_calc(self):
        # # 判断轨迹可行性
        # if self.a_max * self.h < (self.v0**2 - self.v1**2)/2:
        #     print("轨迹不可行")
        #     self.T = 0
        #     return self.T
        # 2. 假设v_max与a_max可达，计算各段时间值
        value2 = 4 * self.h**2 - 4 * self.h * (self.v0 + self.v1) * self.T + 2 * (self.v0**2 + self.v1**2) * self.T**2
        if value2 < 0.:
            print(f"1. 根号内为负，无法计算轨迹")
            self.T = 0
            return self.T
        self.a_lim = (2 * self.h - self.T * (self.v0 + self.v1) +  np.sqrt(value2))/self.T**2
        if self.a_max < self.a_lim:
            self.a_max = self.a_lim
            print(f"a_lim > a_max，无法计算轨迹,提升a_max={self.a_max:.4f}")
        value1 = self.a_max**2 * self.T**2 - 4 * self.a_max * self.h + \
            2 * self.a_max * (self.v0 + self.v1) * self.T - (self.v0 - self.v1)**2
        if value1 < 0.:
            print(f"2. 根号内为负，无法计算轨迹")
            self.T = 0
            return self.T
        self.v_v = 0.5 * (self.v0 + self.v1 + self.a_max * self.T - np.sqrt(value1))
        self.T_a = (self.v_v - self.v0) / self.a_max
        self.T_d = (self.v_v - self.v1) / self.a_max
        # self.T_a = 0 if self.T_a < 0. else self.T_a
        # self.T_d = 0 if self.T_d < 0. else self.T_d
        self.T_v = self.T - self.T_a - self.T_d 
        return self.T

    def get_profile(self, dt):
        t = 0
        t_list = []
        q_list = []
        dq_list = []
        ddq_list = []
        while t < self.T:
            q_t, dq_t, ddq_t = self.get_traj_by_time(t)
            t_list.append(t)
            q_list.append(q_t)
            dq_list.append(dq_t)
            ddq_list.append(ddq_t)
            t += dt
        q_t, dq_t, ddq_t = self.get_traj_by_time(self.T)
        t_list.append(self.T)
        q_list.append(q_t)
        dq_list.append(dq_t)
        ddq_list.append(ddq_t)
        return t_list, q_list, dq_list, ddq_list
    
    def get_traj_by_time(self,t):
        if t >= self.t0 and t <= self.T_a + self.t0:
            q_t = self.q0 + self.v0 * (t - self.t0) + (self.v_v - self.v0) * ((t - self.t0)**2) / (2 * self.T_a)
            dq_t = self.v0 + (self.v_v - self.v0) * (t - self.t0)/self.T_a
            ddq_t = (self.v_v - self.v0)/self.T_a
            # print(f"加加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v0 {self.v0:.4f}, a_lim_a = {self.a_lim_a:.4f}, T_j1 = {self.T_j1:.4f}")
        if t > self.T_a + self.t0 and t <= self.t1 - self.T_d:
            q_t = self.q0 + self.v0 * self.T_a/2 + self.v_v * (t - self.t0 - self.T_a/2)
            dq_t = self.v_v
            ddq_t = 0
            # print(f"恒加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v0 {self.v0:.4f}, a_lim_a = {self.a_lim_a:.4f}, T_j1 = {self.T_j1:.4f}")
        if t > self.t1 - self.T_d and t <= self.t1:
            q_t = self.q1 - self.v1 * (self.t1 - t) - (self.v_v - self.v1) * ((self.t1 - t)**2)/(2 * self.T_d)
            dq_t = self.v1 + (self.v_v - self.v1) * (self.t1 - t)/self.T_d
            ddq_t = -(self.v_v - self.v1)/self.T_d
            # print(f"减加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f}  v_lim {self.v_lim:.4f}, j_min = {self.j_min:.4f}, T_a = {self.T_a:.4f}")

        q_t = self.sigma * q_t 
        dq_t = self.sigma * dq_t
        ddq_t = self.sigma * ddq_t 
        return q_t, dq_t, ddq_t
    
    def print_info(self):
        print(f"print trajectory info:")
        print(f"q0 = {self.q0:.4f}, q1 = {self.q1:.4f}, v0 = {self.v0:.4f}, v1 = {self.v1:.4f}")
        print(f"T = {self.T:.4f}, Ta = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}")
        print(f"v_v = {self.v_v:.4f}, a_max = {self.a_max:.4f}, a_lim = {self.a_lim:.4f}")
        print(f"加速 {self.t0 + self.T_a:.4f}  恒速 {self.t1 - self.T_d:.4f}  减速 {self.T:.4f}")

    def plot_all_trajectories(self, t_list, pos, vel, acc, t_list1=[], pos1=[], vel1=[], acc1=[]):
        # 7. 绘图
        plt.figure(figsize=(12, 8))
        # 位置
        plt.subplot(3, 1, 1)
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
        plt.subplot(3, 1, 2)
        plt.plot(t_list, vel, 'r-', linewidth=1.5, label='Velocity')
        if len(t_list1) > 0:
            plt.plot(t_list1, vel1, 'r--', linewidth=1.5, label=f'Velocity (scale))')
        plt.axhline(self.v_v, color='orange', linestyle='--', alpha=0.5, label=f'Max Vel')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [a.u./s]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 加速度
        plt.subplot(3, 1, 3)
        plt.plot(t_list, acc, 'g-', linewidth=1.5, label='Acceleration')
        if len(t_list1) > 0:
            plt.plot(t_list1, acc1, 'g--', linewidth=1.5, label=f'Acceleration (scale))')
        plt.axhline(self.a_max, color='orange', linestyle='--', alpha=0.5, label=f'Max Acc')
        plt.axhline(-self.a_max, color='purple', linestyle='--', alpha=0.5, label=f'Min Acc')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [a.u./s²]')
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
            'total_time': t_list,
        }

# ======================== 测试用例（反向运动重点验证） ========================
if __name__ == "__main__":
    # 输入v0_in=2 → 校准后v0=-2（因为q1<q0，dir_pos=-1），位置从15→5
    constrain = [0,30,5,2,10,0,5]  # 有恒速段 正向
    constrain = [0,30,5,2,1,0,5]  # 无恒速段 正向

    constrain = [30,0,-5,0,1,0,5]  # 有恒速段 反向
    # constrain = [30,0,-5,0,1,0,12]  # 无恒速段 反向
    planner = TrapezoidalGenerator(7, constrain[0], constrain[1], constrain[2], constrain[3], \
                    constrain[4], constrain[5], constrain[6])
    # 单点轨迹规划
    # total_time = planner.trapezoidal_trajc_calc()
    # planner.print_info()
    # if total_time > 0:
    #     t_list, pos, vel, acc = planner.get_profile(0.01)
    #     # t_list1, pos1, vel1, acc1, jerk1 = planner.get_profile(0.01,3)
    #     # for i in range(len(t_list)):
    #     #     print(f"t = {t_list[i]:.4f}, q = {pos[i]:.4f}, v = {vel[i]:.4f}, a = {acc[i]:.4f}, j = {jerk[i]:.4f}")
    #     planner.plot_all_trajectories(t_list, pos, vel, acc)

    # 多点轨迹规划
    # for i in range()


