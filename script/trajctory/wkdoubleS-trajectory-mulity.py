import math
import matplotlib.pyplot as plt
import numpy as np
import copy



class DoubleSCurveTrajectoryGenerator:
    def __init__(self, dof, q0_in, q1_in, v0_in, v1_in, vmax, vmin, amax, amin, jmax, jmin):
        self.dof = dof
        self.t_list = []
        self.traj_pos = []
        self.traj_vel = []
        self.traj_acc = []
        self.traj_jerk = []
        self.lambda_ = 0.999
        self.max_iter = 1000
        self.reset(q0_in, q1_in, v0_in, v1_in, vmax, vmin, amax, amin, jmax, jmin)
    
    def reset(self, q0_in, q1_in, v0_in, v1_in, vmax, vmin, amax, amin, jmax, jmin):
        # 1. 给定初始条件，做相应转换
        self.sigma = np.array([np.sign(q1_in - q0_in) for q1_in, q0_in in zip(q1_in, q0_in)])
        self.q0 = [q0_in * sigma for q0_in, sigma in zip(q0_in, self.sigma)]
        self.q1 = [q1_in * sigma for q1_in, sigma in zip(q1_in, self.sigma)]
        self.v0 = [v0_in * sigma for v0_in, sigma in zip(v0_in, self.sigma)]
        self.v1 = [v1_in * sigma for v1_in, sigma in zip(v1_in, self.sigma)]
        self.v_max = [0.5 * (sigma + 1) * vmax + 0.5 * (sigma - 1) * vmin for vmax, vmin, sigma in zip(vmax, vmin, self.sigma)]
        self.v_min = [0.5 * (sigma + 1) * vmin + 0.5 * (sigma - 1) * vmax for vmax, vmin, sigma in zip(vmax, vmin, self.sigma)] 
        self.a_max = [0.5 * (sigma + 1) * amax + 0.5 * (sigma - 1) * amin for amax, amin, sigma in zip(amax, amin, self.sigma)]
        self.a_min = [0.5 * (sigma + 1) * amin + 0.5 * (sigma - 1) * amax for amax, amin, sigma in zip(amax, amin, self.sigma)]
        self.j_max = [0.5 * (sigma + 1) * jmax + 0.5 * (sigma - 1) * jmin for jmax, jmin, sigma in zip(jmax, jmin, self.sigma)]
        self.j_min = [0.5 * (sigma + 1) * jmin + 0.5 * (sigma - 1) * jmax for jmax, jmin, sigma in zip(jmax, jmin, self.sigma)]

        self.T = np.zeros(self.dof)
        self.T_a = np.zeros(self.dof)
        self.T_d = np.zeros(self.dof)
        self.T_j1 = np.zeros(self.dof)
        self.T_j2 = np.zeros(self.dof)
        self.T_v = np.zeros(self.dof)
        self.T_j = np.zeros(self.dof)

        self.v_lim = np.zeros(self.dof)
        self.a_lim_a = np.zeros(self.dof)
        self.a_lim_d = np.zeros(self.dof)

    def multi_double_s_curve_trajectory(self, dt):
        max_time = 0.
        for i in range(self.dof):
            total_time = self.double_s_curve_trajectory(axis = i)
            if total_time > max_time:
                max_time = total_time
            # 初始化空列表来收集所有关节的数据
        all_pos = []
        all_vel = []
        all_acc = []
        all_jerk = []
        print(f"max_time: {max_time}")
        # 为每个关节生成轨迹
        list_length = []
        for axis in range(self.dof):
            t_list, pos, vel, acc, jerk = self.get_profile(dt, axis, max_time)
            length = len(pos)
            list_length.append(length)  
            print(f"axis: {axis}, length: {length}")
        
            all_pos.append(pos)
            all_vel.append(vel)
            all_acc.append(acc)
            all_jerk.append(jerk)
            
        # 使用np.array()直接创建二维数组
        self.t_list = t_list
        self.traj_pos = np.array(all_pos)
        self.traj_vel = np.array(all_vel)
        self.traj_acc = np.array(all_acc)
        self.traj_jerk = np.array(all_jerk)
        print(f"traj_pos.shape {self.traj_pos.shape}")
        return max_time
            

    def double_s_curve_trajectory(self, axis):
        # 2. 假设v_max与a_max可达，计算各段时间值
        if (self.v_max[axis] - self.v0[axis]) * self.j_max[axis] < self.a_max[axis] ** 2:
            print(f"无法达到a_max")
            self.T_j1[axis] = np.sqrt((self.v_max[axis] - self.v0[axis]) / self.j_max[axis])
            self.T_a[axis] = 2 * self.T_j1[axis]
        else:
            self.T_j1[axis] = self.a_max[axis] / self.j_max[axis]
            self.T_a[axis] = self.T_j1[axis] + (self.v_max[axis] - self.v0[axis]) / self.a_max[axis]

        if (self.v_max[axis] - self.v1[axis]) * self.j_max[axis] < self.a_max[axis] ** 2:
            print(f"无法达到a_min")
            self.T_j2[axis] = np.sqrt((self.v_max[axis] - self.v1[axis]) / self.j_max[axis])    
            self.T_d[axis] = 2 * self.T_j2[axis]
        else:
            self.T_j2[axis] = self.a_max[axis] / self.j_max[axis]
            self.T_d[axis] = self.T_j2[axis] + (self.v_max[axis] - self.v1[axis]) / self.a_max[axis]
        
        self.T_v[axis] = (self.q1[axis] - self.q0[axis])/self.v_max[axis] - 0.5 * self.T_a[axis] * (1 + self.v0[axis]/self.v_max[axis]) - 0.5 * self.T_d[axis] * (1 + self.v1[axis]/self.v_max[axis])

        # 3. 分情况处理
        if self.T_v[axis] > 0:
            print(f"可以达到v_max")
            self.T[axis] = self.T_a[axis] + self.T_d[axis] + self.T_v[axis]
            print(f"Ta = {self.T_a[axis]:.4f}, Tv = {self.T_v[axis]:.4f}, Td = {self.T_d[axis]:.4f}, Tj1 = {self.T_j1[axis]:.4f}, Tj2 = {self.T_j2[axis]:.4f}")
            self.calc_max_vel_and_acc(axis)
            print(f"calculation finished, total time is {self.T[axis]:.4f}")
            return self.T[axis]
        
        self.T_v[axis] = 0.
        print(f"无法达到v_max")
        self.T[axis] = self.T_a[axis] + self.T_d[axis] + self.T_v[axis]
        print(f"T = {self.T[axis]:.4f} Ta = {self.T_a[axis]:.4f}, Tv = {self.T_v[axis]:.4f}, Td = {self.T_d[axis]:.4f}, Tj1 = {self.T_j1[axis]:.4f}, Tj2 = {self.T_j2[axis]:.4f}")
        iteration = 0
        while True:
            if iteration >= self.max_iter:
                break
            self.calc_traj_para(axis)
            # self.print_info()
            if self.T_a[axis] < 0 or self.T_d[axis] < 0:
                if self.T_a[axis] < 0 and self.v0[axis] > self.v1[axis]:
                    print("不存在加速段")
                    self.T_a[axis] = 0.
                    self.T_d[axis] = 2 * (self.q1[axis] - self.q0[axis])/(self.v1[axis] + self.v0[axis])
                    self.T_j1[axis] = 0.
                    value = self.j_max[axis] * (self.j_max[axis] * ((self.q1[axis] - self.q0[axis])**2) + ((self.v1[axis] + self.v0[axis])**2) * (self.v1[axis] - self.v0[axis]))
                    if value < 0.:
                        print(f"3.28b根号内为负，无法计算轨迹")
                        self.T[axis] = 0
                        break
                    self.T_j2[axis] = (self.j_max[axis] * (self.q1[axis] - self.q0[axis]) - np.sqrt(value))/(self.j_max[axis] * (self.v1[axis] + self.v0[axis]))
                if self.T_d[axis] < 0 and self.v1[axis] > self.v0[axis]:
                    print("不存在减速段")
                    self.T_d[axis] = 0.
                    self.T_a[axis] = 2 * (self.q1[axis] - self.q0[axis])/(self.v1[axis] + self.v0[axis])
                    self.T_j2[axis] = 0.
                    value = self.j_max[axis] * (self.j_max[axis] * (self.q1[axis] - self.q0[axis])**2 - (self.v1[axis] + self.v0[axis])**2 * (self.v1[axis] - self.v0[axis]))
                    if value < 0.:
                        print(f"3.29b根号内为负，无法计算轨迹")
                        self.T[axis] = 0
                        break
                    self.T_j1[axis] = (self.j_max[axis] * (self.q1[axis] - self.q0[axis]) - np.sqrt(value))/(self.j_max[axis] * (self.v1[axis] + self.v0[axis]))
                self.calc_max_vel_and_acc(axis)
                print(f"calculation finished, total time is {self.T[axis]:.4f}")
                return self.T[axis]
            else:
                if self.T_a[axis] > 2 * self.T_j[axis] and self.T_d[axis] > 2 * self.T_j[axis]:
                    print(f"存在加减速段")
                    self.calc_max_vel_and_acc(axis)
                    print(f"calculation finished, total time is {self.T[axis]:.4f}")
                    break
                else:
                    iteration += 1
                    # print(f"iteration {iteration} start")
                    self.a_max[axis] *= self.lambda_
                    self.a_min[axis] *= self.lambda_
        return self.T[axis]
            
    def calc_traj_para(self, axis):
        self.T_j1[axis] = self.T_j2[axis] = self.T_j[axis] = self.a_max[axis] / self.j_max[axis]
        delta = (self.a_max[axis]**4)/(self.j_max[axis]**2) + 2 * (self.v0[axis]**2 + self.v1[axis]**2) + \
        self.a_max[axis] * (4 * (self.q1[axis] - self.q0[axis]) - (2 * self.a_max[axis] * (self.v0[axis] + self.v1[axis]))/self.j_max[axis])
        if delta < 0.:
            print(f"3.27根号内为负，无法计算轨迹")
            self.T[axis] = 0
            return self.T[axis]
        self.T_a[axis] = ((self.a_max[axis]**2)/self.j_max[axis] - 2 * self.v0[axis] + np.sqrt(delta))/(2 * self.a_max[axis])
        self.T_d[axis] = ((self.a_max[axis]**2)/self.j_max[axis] - 2 * self.v1[axis] + np.sqrt(delta))/(2 * self.a_max[axis])
        # print(f"calc_traj_para T_a = {self.T_a:.4f}, Tv = {self.T_v:.4f}, Td = {self.T_d:.4f}, Tj1 = {self.T_j1:.4f}, Tj2 = {self.T_j2:.4f}")
        self.T[axis] = self.T_a[axis] + self.T_d[axis] + self.T_v[axis]

    def calc_max_vel_and_acc(self, axis):
        self.T[axis] = self.T_a[axis] + self.T_d[axis] + self.T_v[axis]
        self.a_lim_a[axis] = self.j_max[axis] * self.T_j1[axis]
        self.a_lim_d[axis] = -self.j_max[axis] * self.T_j2[axis]
        self.v_lim[axis] = self.v0[axis] + (self.T_a[axis] - self.T_j1[axis]) * self.a_lim_a[axis]

    def get_profile(self, dt, axis, T = None):
        factor = 1.
        if T != None:
            factor = self.T[axis] / T
            if factor > 1. or factor <= 0.:
                print(f"factor must be in (0,1]")
                return None
        t = 0
        dt = dt * factor
        t_list = []
        q_list = []
        dq_list = []
        ddq_list = []
        dddq_list = []
        epsilon = 1e-10  # 极小值容差
        while t < self.T[axis] - epsilon:
            q_t, dq_t, ddq_t, dddq_t = self.get_traj_by_time_scale(t, axis, factor)
            t_list.append(t / factor)
            q_list.append(q_t)
            dq_list.append(dq_t)
            ddq_list.append(ddq_t)
            dddq_list.append(dddq_t)
            t += dt
        q_t, dq_t, ddq_t, dddq_t = self.get_traj_by_time_scale(self.T[axis], axis, factor)
        t_list.append(self.T[axis] / factor)
        q_list.append(q_t)
        dq_list.append(dq_t)
        ddq_list.append(ddq_t)
        dddq_list.append(dddq_t)
        return t_list, q_list, dq_list, ddq_list, dddq_list
    
    def get_traj_by_time_scale(self,t, axis, factor):
        if t >= 0 and t <= self.T_j1[axis]:
            q_t = self.q0[axis] + self.v0[axis] * t + (self.j_max[axis] * t**3)/6.
            dq_t = self.v0[axis] + (self.j_max[axis] * t**2)/2.
            ddq_t = self.j_max[axis] * t
            dddq_t = self.j_max[axis]
            # print(f"加加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v0 {self.v0[axis]:.4f}, a_lim_a = {self.a_lim_a[axis]:.4f}, T_j1 = {self.T_j1[axis]:.4f}")
        if t > self.T_j1[axis] and t <= self.T_a[axis] - self.T_j1[axis]:
            q_t = self.q0[axis] + self.v0[axis] * t + self.a_lim_a[axis] * (3 * t**2 - 3 * self.T_j1[axis] * t + self.T_j1[axis]**2)/6.
            dq_t = self.v0[axis] + self.a_lim_a[axis] * (t - self.T_j1[axis]/2.)
            ddq_t = self.a_lim_a[axis]
            dddq_t = 0
            # print(f"恒加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v0 {self.v0[axis]:.4f}, a_lim_a = {self.a_lim_a[axis]:.4f}, T_j1 = {self.T_j1[axis]:.4f}")
        if t > self.T_a[axis] - self.T_j1[axis] and t <= self.T_a[axis]:
            q_t = self.q0[axis] + 0.5 * (self.v_lim[axis] + self.v0[axis]) * self.T_a[axis] - self.v_lim[axis] * (self.T_a[axis] - t) - (self.j_min[axis] * (self.T_a[axis] - t)**3)/6.
            dq_t = self.v_lim[axis] + (self.j_min[axis] * (self.T_a[axis] - t)**2)/2.
            ddq_t = -self.j_min[axis] * (self.T_a[axis] - t)
            dddq_t = self.j_min[axis]
            # print(f"减加速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f}  v_lim {self.v_lim:.4f}, j_min = {self.j_min:.4f}, T_a = {self.T_a:.4f}")
        if t > self.T_a[axis] and t <= self.T_a[axis] + self.T_v[axis]:
            q_t = self.q0[axis] + 0.5 * (self.v_lim[axis] + self.v0[axis]) * self.T_a[axis] + self.v_lim[axis] * (t - self.T_a[axis]) 
            dq_t = self.v_lim[axis] 
            ddq_t = 0
            dddq_t = 0
            # print(f"恒速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v_lim {self.v_lim:.4f}, j_min = {self.j_min:.4f}, T_a = {self.T_a:.4f}")
        if t > self.T[axis] - self.T_d[axis] and t <= self.T[axis] - self.T_d[axis] + self.T_j2[axis]:
            q_t = self.q1[axis] - 0.5 * (self.v_lim[axis] + self.v1[axis]) * self.T_d[axis] + self.v_lim[axis] * (t - self.T[axis] + self.T_d[axis]) - (self.j_max[axis] * (t - self.T[axis] + self.T_d[axis])**3)/6.
            dq_t = self.v_lim[axis] - (self.j_max[axis] * (t - self.T[axis] + self.T_d[axis])**2)/2      
            ddq_t = -self.j_max[axis] * (t - self.T[axis] + self.T_d[axis])
            dddq_t = self.j_min[axis]
            # print(f"加减速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v_lim {self.v_lim[axis]:.4f}, a_lim_d = {self.a_lim_d[axis]:.4f}, j_max = {self.j_max[axis]:.4f}")
        if t > self.T[axis] - self.T_d[axis] + self.T_j2[axis] and t <= self.T[axis] - self.T_j2[axis]:
            q_t = self.q1[axis] - 0.5 * (self.v_lim[axis] + self.v1[axis]) * self.T_d[axis] + self.v_lim[axis] * (t - self.T[axis] + self.T_d[axis]) + \
            (self.a_lim_d[axis]/6.) * (3 * ((t - self.T[axis] + self.T_d[axis])**2) - 3 * self.T_j2[axis] * (t - self.T[axis] + self.T_d[axis]) + self.T_j2[axis]**2)
            dq_t = self.v_lim[axis] + self.a_lim_d[axis] * (t - self.T[axis] + self.T_d[axis] - self.T_j2[axis]/2)
            ddq_t = self.a_lim_d[axis]
            dddq_t = 0
            # print(f"恒减速 t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f}  q1 {self.q1[axis]:.4f}, v1 {self.v1[axis]:.4f}, v_lim {self.v_lim[axis]:.4f}")
        if t > self.T[axis] - self.T_j2[axis] and t <= self.T[axis]:
            q_t = self.q1[axis] - self.v1[axis] * (self.T[axis] - t) - (self.j_max[axis] * (self.T[axis] - t)**3)/6.
            dq_t = self.v1[axis] + (self.j_max[axis] * (self.T[axis] - t)**2)/2.
            ddq_t = -self.j_max[axis] * (self.T[axis] - t)
            dddq_t = self.j_max[axis]
            # print(f"减减速t {t:.4f}, q = {q_t:.4f} dq = {dq_t:.4f} v_lim {self.v_lim:.4f}, j_max = {self.j_max:.4f}, T_d = {self.T_d:.4f}")
        q_t = self.sigma[axis] * q_t 
        dq_t = self.sigma[axis] * dq_t * factor
        ddq_t = self.sigma[axis] * ddq_t * factor**2
        dddq_t = self.sigma[axis] * dddq_t * factor**3
        return q_t, dq_t, ddq_t, dddq_t
    

    
    def print_info(self):
        print(f"print trajectory info:")
        print(f"q0 = {self.q0}, \nq1 = {self.q1}, \nv0 = {self.v0}, \nv1 = {self.v1}\n")
        print(f"T = {self.T},\n Ta = {self.T_a},\n Tv = {self.T_v},\n Td = {self.T_d}, \n\
        T_j = {self.T_j}\n Tj1 = {self.T_j1}, \nTj2 = {self.T_j2}\n")
        print(f"vlim = {self.v_lim},\n a_lim_a = {self.a_lim_a}, \n a_lim_d = {self.a_lim_d}")
        print(f"vmax = {self.v_max}, \n vmin = {self.v_min},\n a_max = {self.a_max}, \n a_min = {self.a_min}, \n jmax = {self.j_max}, \njmin = {self.j_min}")
        print(f"加加速 {self.T_j1}\n 恒加速 {self.T_a - self.T_j1} \n减加速 {self.T_a:} \n恒速 {self.T_a + self.T_v} \
    {self.T - self.T_d} 加减速 {self.T - self.T_d + self.T_j2} 恒减速 {self.T - self.T_j2} 减减速 {self.T}")

    def plot_all_trajectories(self, t_list, pos, vel, acc, jerk, t_list1=[], pos1=[], vel1=[], acc1=[], jerk1=[]):
        print(f"dof {self.dof}")
        # 7. 绘图
        plt.figure(figsize=(12, 8))
        # 位置
        for i in range(self.dof):
            print(f"{i} pos pos.shape {pos[i]}")
            plt.plot(t_list, pos[i], linewidth=1.5, label=f'Position {i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [a.u.]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 速度
        plt.figure(figsize=(12, 8))
        for i in range(self.dof):
            plt.plot(t_list, vel[i], linewidth=1.5, label=f'Velocity {i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [a.u./s]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 加速度
        plt.figure(figsize=(12, 8))
        for i in range(self.dof):
            plt.plot(t_list, acc[i], linewidth=1.5, label=f'Acceleration {i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [a.u./s²]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 加加速度
        plt.figure(figsize=(12, 8))
        for i in range(self.dof):
            plt.plot(t_list, jerk[i], linewidth=1.5, label='Jerk {i}')
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
    constrains = np.array(([0,10,3,-2,5,-5,10,-10,30,-30],
                  [0,10,1,0,5,-5,10,-10,30,-30],
                  [0,13,1,0,10,-10,10,-10,30,-30],
                    [30,10,1,0,10,-10,10,-10,30,-30],
                    [-10,10,1,0,5,-5,10,-10,30,-30],
                    [0,11,1,0,10,-10,20,-20,30,-30],
                    [2,10,1,0,10,-10,10,-10,30,-30]
                    ))

    planner = DoubleSCurveTrajectoryGenerator(7, constrains[:,0], constrains[:,1], constrains[:,2], constrains[:,3], \
                    constrains[:,4], constrains[:,5], constrains[:,6], constrains[:,7], constrains[:,8], constrains[:,9])
    total_time = planner.multi_double_s_curve_trajectory(0.001)
    if total_time > 0:
        t_list = planner.t_list
        pos = planner.traj_pos
        vel = planner.traj_vel
        acc = planner.traj_acc
        jerk = planner.traj_jerk
        print(f"pos {pos.shape} vel {vel.shape} acc {acc.shape} jerk {jerk.shape}")
        planner.plot_all_trajectories(t_list, pos, vel, acc, jerk)
        planner.print_info()


