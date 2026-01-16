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
    

class JointTrajectory:
    def __init__(self, dof, Tf, method = "cubic"):
        self.method = method
        self.coeffs = []  # 存储每个关节的三次多项式系数
        self.dof = dof    # 自由度
        self.Tf = Tf      # 总时间
        self.traj = []    # 存储生成的轨迹
        self.velocy = []  # 存储生成的速度

    def multi_trajectory(self, thetastart, thetaend, v0_arr, vf_arr):
        """生成多关节轨迹"""
        self.coeffs = []  # 清空之前的系数
        for j in range(self.dof):
            if self.method == "cubic":
                # 为每个关节计算三次多项式系数
                self.coeffs.append(self.Cubic_curve_calc(0, self.Tf, 
                                                        thetastart[j], thetaend[j], 
                                                        v0_arr[j], vf_arr[j]))
            if self.method == "quintic":
                # 为每个关节计算五次多项式系数
                self.coeffs.append(self.Quintic_curve_calc(0, self.Tf, 
                                                        thetastart[j], thetaend[j], 
                                                        v0_arr[j], vf_arr[j]))
        # 将系数转换为numpy数组以便后续计算
        self.coeffs = np.array(self.coeffs)

    def get_trajectory(self, num):
        """获取离散轨迹点"""
        self.traj = []  # 清空之前的轨迹
        for j in range(num):
            # 计算当前时间点
            t = j * self.Tf / (num - 1)
            # 计算所有关节在当前时间点的位置
            self.traj.append(self.get_traj_p(t))
        # 转换为numpy数组并返回
        return np.array(self.traj)
    
    def get_velocy(self, num):
        """获取离散速度点"""
        self.velocy = []  # 清空之前的速度
        for j in range(num):
            # 计算当前时间点
            t = j * self.Tf / (num - 1)
            # 计算所有关节在当前时间点的速度
            self.velocy.append(self.get_traj_v(t))
        # 转换为numpy数组并返回
        return np.array(self.velocy)

    def Cubic_curve_calc(self, u0, u1, bgP, endP, bgV, endV):
        """计算三次多项式系数"""
        A = np.array([[1, u0,  u0 ** 2, u0 ** 3],
                    [1, u1,  u1 ** 2, u1 ** 3],
                    [0, 1, 2 * u0, 3 * u0 ** 2],
                    [0, 1, 2 * u1, 3 * u1 ** 2]])
        p = np.array([bgP, endP, bgV, endV])
        return np.linalg.solve(A, p)
    
    def Quintic_curve_calc(self, u0, u1, bgP, endP, bgV, endV, bgA, endA):
        """计算三次多项式系数"""
        A = np.array([[1, u0,  u0 ** 2, u0 ** 3,u0 ** 4,u0 ** 5],
                    [1, u1,  u1 ** 2, u1 ** 3,u1 ** 4,u1 ** 5],
                    [0, 1, 2 * u0, 3 * u0 ** 2, 4 * u0 ** 3, 5 * u0 ** 4],
                    [0, 1, 2 * u1, 3 * u1 ** 2, 4 * u1 ** 3, 5 * u1 ** 4],
                    [0, 0, 2, 6 * u0, 12 * u0 ** 2, 20 * u0 ** 3],
                    [0, 0, 2, 6 * u1, 12 * u1 ** 2, 20 * u1 ** 3]])
        p = np.array([bgP, endP, bgV, endV, bgA, endA])
        return np.linalg.solve(A, p)

    def get_traj_p(self, t):
        """根据时间t计算所有关节的位置"""
        if self.method == "cubic":
            # 计算三次多项式的基函数
            basis = np.array([1, t, t ** 2, t ** 3])
        if self.method == "quintic":
            # 计算五次多项式的基函数
            basis = np.array([1, t, t ** 2, t ** 3, t ** 4, t ** 5])
        # 计算所有关节的位置：系数矩阵与基函数的点积
        return np.dot(self.coeffs, basis)
    
    def get_traj_v(self, t):
        """根据时间t计算所有关节的位置"""
        if self.method == "cubic":
            # 计算三次多项式的基函数
            basis = np.array([0, 1, 2 * t, 3 * t ** 2])
        if self.method == "quintic":
            # 计算五次多项式的基函数
            basis = np.array([0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4])
        # 计算所有关节的位置：系数矩阵与基函数的点积
        return np.dot(self.coeffs, basis)


# 设置numpy输出格式
np.set_printoptions(precision=4, suppress=True)

if __name__ == '__main__':
    dt = 50 
    N = 100
    delay_count = 3
    # 创建轨迹规划器实例
    planner = JointTrajectory(7, dt, method="cubic")  # 4个自由度，总时间5秒
    # 测试用例1 
    # theta = np.array(([0, -1],
    #                     [1, -2],
    #                     [3, 2],
    #                     [2, 4],
    #                     [-1, 6],
    #                     [-3, 3],
    #                     [2,3]))
    # traj_list = []
    # velocity_list = []
    # last_p = theta[0, :]
    # last_v = 0.5 * (theta[1, :] - last_p) / dt
    # print(f"theta.shape[0] = {theta.shape[0]}, theta[i, :] = {theta[0, :]}")
    # for i in range(theta.shape[0]):
    #     if i == 0:
    #         continue
    #     cur_p = theta[i, :]
    #     if i == theta.shape[0]-1:
    #         next_p = theta[i, :]
    #     else:
    #         next_p = theta[i+1, :]
            
    #     cur_v = 0.5 * (next_p - last_p) / dt
    #     print(f"cur_v {i} = {cur_v} last_v {last_v}")
    #     planner.multi_trajectory(last_p, cur_p, last_v, cur_v)
    #     traj = planner.get_trajectory(N)
    #     velocity = planner.get_velocy(N)
    #     traj_list.append(traj)
    #     velocity_list.append(velocity)
    #     last_p = cur_p
    #     last_v = cur_v
    # # 将列表转换为numpy数组
    # traj_list = np.concatenate(traj_list, axis=0)
    # velocity_list = np.concatenate(velocity_list, axis=0)
    # # print(velocity_list[500:])
    # plot_trajectory(traj_list, velocity_list, dt, N)

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
    print(f"positions.shape[0] = {positions.shape[0]}, positions[i, :] = {positions[0, :]}")
    for i in range(positions.shape[0]):
        if i == 0:
            continue
        cur_p = positions[i, :]
        if i == positions.shape[0]-1:
            next_p = positions[i, :]
        else:
            next_p = positions[i+1, :]
            
        cur_v = 0.5 * (next_p - last_p) / dt
        for i in range(7):
            if (next_p[i] - cur_p[i]) *  (cur_p[i] - last_p[i]) < 0:
                cur_v[i] = 0
                # print(f"cur_v {i} = {cur_v[i]} last_v {last_v[i]}")
        # print(f"cur_v {i} = {cur_v} last_v {last_v}")
        planner.multi_trajectory(last_p, cur_p, last_v, cur_v)
        traj = planner.get_trajectory(N)
        velocity = planner.get_velocy(N)
        traj_list.append(traj)
        velocity_list.append(velocity)
        last_p = cur_p
        last_v = cur_v
    # 将列表转换为numpy数组
    traj_list = np.concatenate(traj_list, axis=0)
    velocity_list = np.concatenate(velocity_list, axis=0)
    # print(velocity_list[500:])
    # plot_trajectory(traj_list, velocity_list, dt, N)
    # 绘制图形
    print("\n正在生成图形...")
    show_range = []
    plot_all_trajectories(positions, traj_list, velocity_list, dt, N, delay_count, show_range)


        
