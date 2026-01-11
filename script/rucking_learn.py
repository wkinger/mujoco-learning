import ruckig
import numpy as np
import matplotlib.pyplot as plt

# 维度
DOFs = 7  # 7自由度机械臂

# 创建 Ruckig 实例
otg = ruckig.Ruckig(DOFs, 0.01)  # 时间步长为 0.01 秒

# 输入和输出参数
input = ruckig.InputParameter(DOFs)
output = ruckig.OutputParameter(DOFs)

# 设置初始状态
input.current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
input.current_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
input.current_acceleration = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 设置目标状态（假设已通过逆运动学计算得到）
input.target_position = [1.0, 0.5, 0.25, 0.0, -1.0, -0.5, -0.25]
input.target_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
input.target_acceleration = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 设置最大速度、加速度和加加速度
input.max_velocity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
input.max_acceleration = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
input.max_jerk = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# 用于存储轨迹数据
positions = []
velocities = []
accelerations = []

# 计算轨迹
while True:
    result = otg.update(input, output)

    if result == ruckig.Result.Finished:
        break

    # 存储当前状态
    positions.append(output.new_position.copy())
    velocities.append(output.new_velocity.copy())
    accelerations.append(output.new_acceleration.copy())

    # 更新输入参数
    input.current_position = output.new_position
    input.current_velocity = output.new_velocity
    input.current_acceleration = output.new_acceleration

# 转换为 NumPy 数组
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)

# 绘制轨迹图
time_steps = np.arange(positions.shape[0]) * 0.01

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

for i in range(DOFs):
    axs[0].plot(time_steps, positions[:, i], label=f'Joint {i+1}')
    axs[1].plot(time_steps, velocities[:, i], label=f'Joint {i+1}')
    axs[2].plot(time_steps, accelerations[:, i], label=f'Joint {i+1}')

axs[0].set_title('Position')
axs[1].set_title('Velocity')
axs[2].set_title('Acceleration')

for ax in axs:
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()
