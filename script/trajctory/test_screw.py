import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
# 测试现代机器人学库代码
def plot_trajectory_analysis(traj):
    """
    绘制轨迹的位移、速度和加速度曲线图
    
    参数:
    traj: np.array类型的轨迹数据，每行包含n+1个元素，其中前n个是轨迹变量
    
    返回:
    None
    """
    # 设置numpy输出格式：3位浮点数，不使用科学计数法
    np.set_printoptions(precision=3, suppress=True)
    
    # 确保输入是numpy数组
    traj = np.array(traj)
    
    # 获取轨迹维度（假设每行最后一个元素是时间戳或其他非轨迹数据，所以维度是列数-1）
    dim = traj.shape[1] 
    
    # 生成时间序列（假设每行对应一个时间点，间隔为1秒）
    t = np.arange(len(traj))  # 时间点：0, 1, 2, ..., len(traj)-1
    
    # 提取位移数据
    displacement = traj[:, 0:dim]
    
    # 计算速度（一阶差分）
    dt = np.diff(t)  # 时间间隔
    velocity = np.diff(displacement, axis=0) / dt[:, np.newaxis]  # 利用广播机制计算速度
    
    # 计算加速度（二阶差分）
    acceleration = np.diff(velocity, axis=0) / dt[1:, np.newaxis]  # 利用广播机制计算加速度
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 绘制位移曲线
    plt.subplot(3, 1, 1)
    for i in range(dim):
        plt.plot(t, displacement[:, i], marker='o', linewidth=2, label=f'Var {i+1}')
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
        plt.plot(t[:-1], velocity[:, i], marker='s', linewidth=2, label=f'Var {i+1}')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Velocity', fontsize=12)
    plt.title('Velocity Curves', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 绘制加速度曲线
    plt.subplot(3, 1, 3)
    for i in range(dim):
        plt.plot(t[1:-1], acceleration[:, i], marker='^', linewidth=2, label=f'Var {i+1}')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Acceleration', fontsize=12)
    plt.title('Acceleration Curves', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()  # 调整子图间距
    plt.show()


np.set_printoptions(precision=4, suppress=True)
if __name__ == '__main__':
    d2 = 263 * 0.001
    d3 = 292 * 0.001
    a = 17 * 0.001
    M = np.array([[1, 0,  0, d2 + d3],
                [ 0, 0,  1, 0],
                [ 0, -1, 0, 0],
                [ 0, 0,  0, 1]])
    Slist = np.array([[0, 0, 1, 0, 0,   0],
                    [0, 1,  0, 0, 0,   0],
                    [1, 0,  0, 0, 0,   0],
                    [0, 0,  1, a, -d2, 0],
                    [1, 0,  0, 0, 0,   0],
                    [0, 0,  1, 0, -d2 - d3, 0],
                    [0, 1,  0, 0, 0, d2 + d3]]).T
    thetalist = np.array([6, -15, 30, 33, -14, -27, -12])
    thetalist1 = np.array([0,0,0,0,0,0,0])

    # for i in range(7):
    #     thetalist1[i] = thetalist[i]
    #     print(f"thetalist1 = {thetalist1}")
    #     thetalist2 = np.deg2rad(thetalist1.copy())
    #     T = mr.FKinSpace(M, Slist, thetalist2.copy())
    #     print(f"T = {T}")
    #     eomg = 0.01
    #     ev = 0.001
    #     joint = mr.IKinSpace(Slist, M, T, np.deg2rad(np.array([9, -19, 36, 30, -15, -7, -2])), eomg, ev)
    #     print(f"joint = {np.rad2deg(joint[0])}\n")

# JointTrajectory
    # thetastart = np.array([-1, -1,  -1, -1, 3])
    # thetaend = np.array([2,  3, 1.1, 2, 0.9])
    # Tf = 4
    # N = 100
    # method = 5
    # traj = mr.JointTrajectory(thetastart, thetaend, Tf, N, method)
    # print(f"traj = {traj}")
    # # 调用函数绘制轨迹分析图
    # plot_trajectory_analysis(traj)
# ScrewTrajectory
    Xstart = np.array([[1, 0, 0, 1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]])
    Xend = np.array([[0, 0, 1, 0.1],
                        [1, 0, 0,   0],
                        [0, 1, 0, 4.1],
                        [0, 0, 0,   1]])
    Tf = 5
    N = 4
    method = 3
    traj = mr.ScrewTrajectory(Xstart, Xend, Tf, N, method)
    print(f"traj = {traj}")
    # 调用函数绘制轨迹分析图
    plot_trajectory_analysis(traj)