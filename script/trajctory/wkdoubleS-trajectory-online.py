import math
import matplotlib.pyplot as plt
import numpy as np
import copy

epsilon = 1e-10  # 极小值容差


class DoubleSCurveTrajectoryGenerator:
    def __init__(self, dof, T_s, q0_in, q1_in, v0_in, v1_in, a0_in, a1_in, vmax, vmin, amax, amin, jmax, jmin):
        self.dof = dof
        self.T_s = T_s
        self.still = False
        self.reset(q0_in, q1_in, v0_in, v1_in, a0_in, a1_in, vmax, vmin, amax, amin, jmax, jmin)
    
    def reset(self, q0_in, q1_in, v0_in, v1_in, a0_in, a1_in, vmax, vmin, amax, amin, jmax, jmin):
        # 1. 给定初始条件，做相应转换
        self.sigma = np.sign(q1_in - q0_in)
        if abs(self.sigma) < epsilon:
            self.sigma = 1
        self.q0 = q0_in * self.sigma
        self.q1 = q1_in * self.sigma
        self.v0 = v0_in * self.sigma
        self.v1 = v1_in * self.sigma
        self.a0 = a0_in * self.sigma
        self.a1 = a1_in * self.sigma
        self.v_max = 0.5 * (self.sigma + 1) * vmax + 0.5 * (self.sigma - 1) * vmin
        self.v_min = 0.5 * (self.sigma + 1) * vmin + 0.5 * (self.sigma - 1) * vmax
        self.a_max = 0.5 * (self.sigma + 1) * amax + 0.5 * (self.sigma - 1) * amin
        self.a_min = 0.5 * (self.sigma + 1) * amin + 0.5 * (self.sigma - 1) * amax
        self.j_max = 0.5 * (self.sigma + 1) * jmax + 0.5 * (self.sigma - 1) * jmin
        self.j_min = 0.5 * (self.sigma + 1) * jmin + 0.5 * (self.sigma - 1) * jmax

        self.T = 0.
        self.T_j2a = 0.
        self.T_j2b = 0.

        self.v_lim = 0.
        self.a_lim_a = 0.
        self.a_lim_d = 0.

        self.q_k = 0.
        self.dq_k = 0.
        self.ddq_k = 0.
        self.dddq_k = 0.
        self.q_k_last = self.q0
        self.dq_k_last = self.v0
        self.ddq_k_last = self.a0
        self.dddq_k_last = 0.

        self.h_k = 0.
        self.time_cur = 0.
        self.time_stamp = 0
        self.time_stage2 = 0.
        self.time_max_vel = 0.
        self.start_stage2 = False
        self.reach_max_vel = False
        self.still = False
        self.stop = False

    def double_s_curve_trajectory(self):
        if abs(self.q0 - self.q1) < epsilon:
            print(f"q0 and q1 are the same")
            self.still = True
            return self.T
        if not self.start_stage2:
            self.T_j2a = (self.a_min - self.ddq_k)/self.j_min
            self.T_j2b = (self.a0 - self.a_min)/self.j_max
            self.T_d = (self.v1 - self.dq_k)/self.a_min + self.T_j2a * (self.a_min - self.ddq_k)/(2 * self.a_min) \
                + self.T_j2b * (self.a_min - self.a1)/(2 * self.a_min)
        if self.T_d < self.T_j2a + self.T_j2b:
            print("amin 不可达")
            value = (self.j_max - self.j_min) * ((self.ddq_k**2) * self.j_max \
                - self.j_min * (self.a1**2 + 2 * self.j_max * (self.dq_k - self.v1)))
            self.T_j2a = -(self.ddq_k/self.j_min) + np.sqrt(value)/(self.j_min * (self.j_min - self.j_max))
            self.T_j2b = self.a1/self.j_max + np.sqrt(value)/(self.j_max * (self.j_max - self.j_min))
            self.T_d = self.T_j2a + self.T_j2b
        if not self.start_stage2:
           self.h_k = (self.ddq_k * self.T_d**2)/2 + (self.j_min * self.T_j2a * (3 * self.T_d**2 \
            - 3 * self.T_j2a * self.T_d + self.T_j2a**2) + self.j_max * self.T_j2b**3)/6 + self.T_d * self.dq_k
        if self.h_k < self.q1 - self.q_k:
            crition = self.dq_k - (self.ddq_k**2)/(2 * self.j_min)
            if crition < self.v_max and self.ddq_k < self.a_max and self.reach_max_vel == False:
                self.dddq_k = self.j_max
            if crition < self.v_max and self.ddq_k >= self.a_max:
                self.dddq_k = 0
            if crition >= self.v_max and self.ddq_k > 0:
                self.dddq_k = self.j_min
            if crition >= self.v_max and self.ddq_k <= 0:
                self.dddq_k = 0
                if not self.reach_max_vel:
                    self.time_max_vel = self.time_stamp
                    self.reach_max_vel = True
                print("reach max vel")
            print(f"第一阶段 crition {crition:.4f} hk {self.h_k:.4f}  q1 - qk {self.q1 - self.q_k:.4f} q1 {self.q1:.4f} qk {self.q_k:.4f}")
        else:
            if self.start_stage2 == False:
                self.time_stage2 = self.time_stamp
                self.start_stage2 = True
            time_stage2 = (self.time_stamp - self.time_stage2) * self.T_s
            if time_stage2 >= 0 and time_stage2 < self.T_j2a:
                self.dddq_k = self.j_min
            if time_stage2 >= self.T_j2a and time_stage2 < (self.T_d - self.T_j2b):
                self.dddq_k = 0
            if time_stage2 >= (self.T_d - self.T_j2b) and time_stage2 < self.T_d:
                self.dddq_k = self.j_max
            if time_stage2 > self.T_d - epsilon:
                self.stop = True
            print(f"第二阶段 time_stamp {self.time_stamp:.4f} time_stage2 {time_stage2:.4f} hk {self.h_k:.4f} q1-qk {self.q1 - self.q_k:.4f} q1 {self.q1:.4f} qk {self.q_k:.4f}")
        print(f"Td {self.T_d:.4f} Tj2b {self.T_j2b:.4f} Tj2a {self.T_j2a:.4f}")
        # 更新轨迹
        self.ddq_k = self.ddq_k_last + self.T_s * (self.dddq_k_last + self.dddq_k)/2
        self.dq_k = self.dq_k_last + self.T_s * (self.ddq_k_last + self.ddq_k)/2
        self.q_k = self.q_k_last + self.T_s * (self.dq_k_last + self.dq_k)/2
        self.dddq_k_last = self.dddq_k
        self.ddq_k_last = self.ddq_k
        self.dq_k_last = self.dq_k
        self.q_k_last = self.q_k
        self.time_stamp += 1
        self.time_cur += self.T_s
        return self.time_cur, self.q_k, self.dq_k, self.ddq_k, self.dddq_k
    
    def print_info(self):
        print(f"print trajectory info:")
        print(f"q0 = {self.q0:.4f}, q1 = {self.q1:.4f}, v0 = {self.v0:.4f}, v1 = {self.v1:.4f} a0 = {self.a0:.4f} a1 = {self.a1:.4f}")
        print(f"T = {self.T:.4f}, Td = {self.T_d:.4f}, Tj2a = {self.T_j2a:.4f}, Tj2b = {self.T_j2b:.4f} \
              time_max_vel {self.time_max_vel * self.T_s:.4f} time_stage2 {self.time_stage2 * self.T_s:.4f}")
        print(f"vmax = {self.v_max:.4f}, vmin = {self.v_min:.4f}, a_max = {self.a_max:.4f}, a_min = {self.a_min:.4f}, jmax = {self.j_max:.4f}, jmin = {self.j_min:.4f}")

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

# ======================== 测试用例（反向运动重点验证） ========================
if __name__ == "__main__":
    # 输入v0_in=2 → 校准后v0=-2（因为q1<q0，dir_pos=-1），位置从15→5
    constrain = [0,20.,0,0,1,0,5,-5,10,-8,30,-40]  # 有恒速段 正向

    dof = 7
    T_s = 0.001
    planner = DoubleSCurveTrajectoryGenerator(dof, T_s, constrain[0], constrain[1], constrain[2], constrain[3], \
                    constrain[4], constrain[5], constrain[6], constrain[7], constrain[8], constrain[9], constrain[10], constrain[11])
    count = 0
    t_list = []
    q_list = []
    dq_list = []
    ddq_list = []
    dddq_list = []
    while planner.stop == False:
        time_cur, q_k, dq_k, ddq_k, dddq_k = planner.double_s_curve_trajectory()
        print(f"{count} time: {time_cur:.4f}, q: {q_k:.4f}, dq: {dq_k:.4f}, ddq: {ddq_k:.4f}, dddq: {dddq_k:.4f}\n")
        if q_k >= constrain[1]:
            break
        t_list.append(time_cur)
        q_list.append(q_k)
        dq_list.append(dq_k)
        ddq_list.append(ddq_k)
        dddq_list.append(dddq_k)
        count += 1
    planner.print_info()

    planner.plot_all_trajectories(t_list, q_list, dq_list, ddq_list, dddq_list)


