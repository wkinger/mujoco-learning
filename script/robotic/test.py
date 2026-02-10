import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 跟踪微分器定义（一阶，适配关节位置） ----------------------
class FirstOrderTD:
    """Single-joint first-order tracking differentiator (outputs smooth position + velocity)"""
    def __init__(self, r=10.0, dt=0.001):
        self.r = r          # Tracking speed coefficient (tunable per joint)
        self.dt = dt        # Control cycle (consistent with robot arm)
        self.x1 = 0.0       # Tracked position
        self.x2 = 0.0       # Extracted velocity

    def reset(self):
        self.x1, self.x2 = 0.0, 0.0

    def update(self, target):
        """Input: single-joint target position; Output: smooth position, smooth velocity"""
        self.x1 += self.r * (target - self.x1) * self.dt
        self.x2 = self.r * (target - self.x1)
        return self.x1, self.x2

# ---------------------- 2. 关节PID控制器定义 ----------------------
class JointPID:
    """Single-joint position-velocity PID controller"""
    def __init__(self, kp=5.0, ki=0.1, kd=0.2, dt=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.error_sum = 0.0  # Integral term
        self.last_error = 0.0  # Last position error

    def reset(self):
        self.error_sum = 0.0
        self.last_error = 0.0

    def compute(self, target_pos, target_vel, current_pos, current_vel):
        """
        Input:
            target_pos: TD output smooth target position
            target_vel: TD output smooth target velocity
            current_pos: Joint actual position
            current_vel: Joint actual velocity
        Output: Joint control torque/voltage
        """
        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel

        p_term = self.kp * pos_error
        i_term = self.ki * self.error_sum
        d_term = self.kd * vel_error

        # Integral clamping to prevent integral saturation
        self.error_sum += pos_error * self.dt
        self.error_sum = np.clip(self.error_sum, -1.0, 1.0)

        output = p_term + i_term + d_term
        self.last_error = pos_error

        return output

# ---------------------- 3. 7-axis manipulator simulation test ----------------------
if __name__ == "__main__":
    # Simulation parameters
    dt = 0.001  # Control cycle (1ms is common for 7-axis manipulators)
    sim_time = 5.0  # Total simulation time
    t = np.arange(0, sim_time, dt)

    # Initialize TD and PID for 7 joints (independent parameters for each joint)
    joint_tds = []
    joint_pids = []
    # Parameter list for 7 joints: r decreases, kp decreases, kd increases (adapts to inertia difference)
    td_r_list = [8, 7, 7, 6, 6, 5, 5]
    pid_kp_list = [10, 9, 9, 8, 8, 7, 7]
    pid_kd_list = [0.1, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25]

    for i in range(7):
        joint_tds.append(FirstOrderTD(r=td_r_list[i], dt=dt))
        joint_pids.append(JointPID(
            kp=pid_kp_list[i],
            ki=0.05,
            kd=pid_kd_list[i],
            dt=dt
        ))

    # Generate target joint positions (step input at 1s to simulate command sending)
    target_joint_pos = np.zeros((7, len(t)))
    for i in range(7):
        step_amp = 0.5 + i * 0.1  # Joint 1: 0.5rad, Joint 7: 1.1rad
        target_joint_pos[i] = np.where(t < 1.0, 0.0, step_amp)

    # Simulation operation
    current_joint_pos = np.zeros((7, len(t)))
    current_joint_vel = np.zeros((7, len(t)))
    control_output = np.zeros((7, len(t)))

    for t_idx in range(len(t)):
        for joint_idx in range(7):
            tar_pos = target_joint_pos[joint_idx, t_idx]
            smooth_pos, smooth_vel = joint_tds[joint_idx].update(tar_pos)

            curr_pos = current_joint_pos[joint_idx, t_idx-1] if t_idx>0 else 0.0
            curr_vel = current_joint_vel[joint_idx, t_idx-1] if t_idx>0 else 0.0
            ctrl_out = joint_pids[joint_idx].compute(
                target_pos=smooth_pos,
                target_vel=smooth_vel,
                current_pos=curr_pos,
                current_vel=curr_vel
            )

            # Simplified joint dynamics: Torque → Acceleration → Velocity → Position
            J = 1.0 - joint_idx * 0.1  # Joint inertia: Joint 1 > Joint 7
            acc = ctrl_out / J
            new_vel = curr_vel + acc * dt
            new_pos = curr_pos + new_vel * dt

            current_joint_pos[joint_idx, t_idx] = new_pos
            current_joint_vel[joint_idx, t_idx] = new_vel
            control_output[joint_idx, t_idx] = ctrl_out

    # ---------------------- 4. Plotting with English annotations ----------------------
    plt.figure(figsize=(12, 8))
    # Joint 1 position tracking
    plt.subplot(2,2,1)
    plt.plot(t, target_joint_pos[0], 'k--', label='Target Position')
    plt.plot(t, current_joint_pos[0], 'r-', label='Actual Position')
    plt.title('Joint 1 Position Tracking (Large Inertia)')
    plt.ylabel('Position (rad)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Joint 7 position tracking
    plt.subplot(2,2,2)
    plt.plot(t, target_joint_pos[6], 'k--', label='Target Position')
    plt.plot(t, current_joint_pos[6], 'b-', label='Actual Position')
    plt.title('Joint 7 Position Tracking (Small Inertia)')
    plt.ylabel('Position (rad)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Joint 1 actual velocity
    plt.subplot(2,2,3)
    plt.plot(t, current_joint_vel[0], 'r-', linewidth=1.5)
    plt.title('Joint 1 Actual Velocity')
    plt.ylabel('Velocity (rad/s)')
    plt.xlabel('Time (s)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Joint 7 controller output
    plt.subplot(2,2,4)
    plt.plot(t, control_output[6], 'b-', linewidth=1.5)
    plt.title('Joint 7 Controller Output Torque')
    plt.ylabel('Control Torque (N·m)')
    plt.xlabel('Time (s)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
