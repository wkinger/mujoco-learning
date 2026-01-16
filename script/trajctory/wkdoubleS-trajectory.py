import math
import matplotlib.pyplot as plt
import numpy as np




class DoubleSCurveTrajectoryGenerator:
    def __init__(self, dof, q0_in, q1_in, v0_in, v1_in, vmax, vmin, amax, amin, jmax, jmin):
        self.dof = dof
        self.traj = []
        self.lambda_ = 0.99
        # 1. 给定初始条件，做相应转换
        self.sigma = np.sign(q1_in - q0_in)
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

    def double_s_curve_trajectory(self):
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
            print(f"无法达到v_max")
            self.T = self.T_a + self.T_d + self.T_v
            print(f"Ta = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}, Tj1 = {self.T_j1:.4f}, Tj2 = {self.T_j2:.4f}")
            self.calc_max_vel_and_acc()
            print(f"calculation finished, total time is {self.T}")
            return self.T
        self.T_v = 0.
        print(f"v_max is not reached")
        self.T = self.T_a + self.T_d + self.T_v
        print(f"T = {self.T:.4f} Ta = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}, Tj1 = {self.T_j1:.4f}, Tj2 = {self.T_j2:.4f}")
        iteration = 0
        while True:
            self.calc_traj_para()
            self.print_info()
            if self.T_a < 0 or self.T_d < 0:
                if self.T_a < 0 and self.v0 > self.v1:
                    print("不存在加速段")
                    self.T_a = 0.
                    self.T_d = 2 * (self.q1 - self.q0)/(self.v1 + self.v0)
                    self.T_j1 = 0.
                    self.T_j2 = (self.j_max * (self.q1 - self.q0) - np.sqrt(self.j_max * (self.j_max * ((self.q1 - self.q0)**2) + ((self.v1 + self.v0)**2) * (self.v1 - self.v0))))/(self.j_max * (self.v1 + self.v0))
                if self.T_d < 0 and self.v1 > self.v0:
                    print("不存在减速段")
                    self.T_d = 0.
                    self.T_a = 2 * (self.q1 - self.q0)/(self.v1 + self.v0)
                    self.T_j2 = 0.
                    self.T_j1 = (self.j_max * (self.q1 - self.q0) - \
                    np.sqrt(self.j_max * (self.j_max * (self.q1 - self.q0)**2 - (self.v1 + self.v0)**2 * (self.v1 - self.v0))))/ \
                    (self.j_max * (self.v1 + self.v0))
                self.calc_max_vel_and_acc()
                print(f"calculation finished, total time is {self.T}")
                return self.T
            else:
                if self.T_a > 2 * self.T_j and self.T_d > 2 * self.T_j:
                    print(f"存在加减速段")
                    self.calc_max_vel_and_acc()
                    print(f"calculation finished, total time is {self.T}")
                    return self.T
                else:
                    iteration += 1
                    print(f"iteration {iteration} start")
                    self.a_max *= self.lambda_
                    self.a_min *= self.lambda_
            
    def calc_traj_para(self):
        self.T_j1 = self.T_j2 = self.T_j = self.a_max / self.j_max
        delta = (self.a_max**4)/(self.j_max**2) + 2 * (self.v0**2 + self.v1**2) + \
        self.a_max * (4 * (self.q1 - self.q0) - (2 * self.a_max * (self.v0 + self.v1))/self.j_max)
        self.T_a = ((self.a_max**2)/self.j_max - 2 * self.v0 + np.sqrt(delta))/(2 * self.a_max)
        self.T_d = ((self.a_max**2)/self.j_max - 2 * self.v1 + np.sqrt(delta))/(2 * self.a_max)
        # print(f"calc_traj_para T_a = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}, Tj1 = {self.T_j1:.4f}, Tj2 = {self.T_j2:.4f}")
        self.T = self.T_a + self.T_d + self.T_v

    def calc_max_vel_and_acc(self):
        self.T = self.T_a + self.T_d + self.T_v
        self.a_lim_a = self.j_max * self.T_j1
        self.a_lim_d = -self.j_max * self.T_j2
        self.v_lim = self.v0 + (self.T_a - self.T_j1) * self.a_lim_a

    def get_profile(self, dt):
        t = 0
        t_list = []
        q_list = []
        dq_list = []
        ddq_list = []
        dddq_list = []
        while t < self.T:
            q_t, dq_t, ddq_t, dddq_t = self.get_traj_by_time(t)
            t_list.append(t)
            q_list.append(q_t)
            dq_list.append(dq_t)
            ddq_list.append(ddq_t)
            dddq_list.append(dddq_t)
            t += dt
        q_t, dq_t, ddq_t, dddq_t = self.get_traj_by_time(self.T)
        t_list.append(self.T)
        q_list.append(q_t)
        dq_list.append(dq_t)
        ddq_list.append(ddq_t)
        dddq_list.append(dddq_t)
        return t_list, q_list, dq_list, ddq_list, dddq_list
    
    def get_traj_by_time(self,t):
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
        dq_t = self.sigma * dq_t
        ddq_t = self.sigma * ddq_t
        dddq_t = self.sigma * dddq_t
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

    def plot_all_trajectories(self, t_list, pos, vel, acc, jerk):
        # 7. 绘图
        plt.figure(figsize=(12, 8))
        # 位置
        plt.subplot(4, 1, 1)
        plt.plot(t_list, pos, 'b-', linewidth=1.5, label=f'Position')
        plt.axhline(pos[0], color='b', linestyle='--', alpha=0.5, label='Start Pos')
        plt.axhline(pos[-1], color='r', linestyle='--', alpha=0.5, label='Target Pos')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [a.u.]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 速度
        plt.subplot(4, 1, 2)
        plt.plot(t_list, vel, 'r-', linewidth=1.5, label='Velocity')
        plt.axhline(self.v_lim, color='orange', linestyle='--', alpha=0.5, label=f'Max Vel')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [a.u./s]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 加速度
        plt.subplot(4, 1, 3)
        plt.plot(t_list, acc, 'g-', linewidth=1.5, label='Acceleration')
        plt.axhline(self.a_max, color='orange', linestyle='--', alpha=0.5, label=f'Max Acc')
        plt.axhline(self.a_min, color='purple', linestyle='--', alpha=0.5, label=f'Min Acc')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [a.u./s²]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 加加速度
        plt.subplot(4, 1, 4)
        plt.plot(t_list, jerk, 'm-', linewidth=1.5, label='Jerk')
        plt.axhline(self.j_max, color='orange', linestyle='--', alpha=0.5, label=f'Max Jerk')
        plt.axhline(self.j_min, color='purple', linestyle='--', alpha=0.5, label=f'Min Jerk')
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
    constrain = [0,10,1,0,5,-5,10,-10,30,-30]  # 有恒速段 正向
    constrain = [10,0,-1,0,5,-5,10,-10,30,-30]  # 有恒速段 反向
    constrain = [0,10,-3,0,5,-5,10,-10,30,-30]  # 有恒速段 正向 反速度 
    constrain = [10,0,3,0,5,-5,10,-10,30,-30]  # 有恒速段 反向 反速度 


    # constrain = [0,10,1,0,10,-10,10,-10,30,-30] # 无恒速段 正向
    # constrain = [10,0,-1,0,10,-10,10,-10,30,-30] # 无恒速段 反向  
    constrain = [0,10,-2,0,10,-10,10,-10,30,-30] # 无恒速段 正向 反速度
    # constrain = [10,0,2,0,10,-10,10,-10,30,-30] # 无恒速段 反向 反速度


    # constrain = [0,10,7,0,10,-10,10,-10,30,-30] # 无恒加速段 正向
    # constrain = [10,0,-7,0,10,-10,10,-10,30,-30] # 无恒加速段 反向
    constrain = [0,10,-2,0,10,-10,20,-20,30,-30] # 无恒加速段 正向 反速度
    # constrain = [10,0,2,0,10,-10,20,-20,30,-30] # 无恒加速段 反向 反速度

    # constrain = [0,10,7.5,0,10,-10,10,-10,30,-30] # 仅有减速段
    # constrain = [0,10,0,7.5,10,-10,10,-10,30,-30] # 仅有加速段
    constrain = [10,0,0,-7.5,10,-10,10,-10,30,-30] # 仅有减速段 反向
    # constrain = [10,0,-7.5,0,10,-10,10,-10,30,-30] # 仅有加速段 正向
    # constrain = [10,0,-7.5,0,10,-10,10,-10,30,-30] # 仅有加速段 正向
    constrain = [0,10,3,10,10,-10,10,-10,30,-30] # 仅有加速段 正向 反速度



    planner = DoubleSCurveTrajectoryGenerator(7, constrain[0], constrain[1], constrain[2], constrain[3], \
                    constrain[4], constrain[5], constrain[6], constrain[7], constrain[8], constrain[9])
    total_time = planner.double_s_curve_trajectory()
    planner.print_info()
        
    t_list, pos, vel, acc, jerk = planner.get_profile(0.01)
    # for i in range(len(t_list)):
    #     print(f"t = {t_list[i]:.4f}, q = {pos[i]:.4f}, v = {vel[i]:.4f}, a = {acc[i]:.4f}, j = {jerk[i]:.4f}")
    planner.plot_all_trajectories(t_list, pos, vel, acc, jerk)


