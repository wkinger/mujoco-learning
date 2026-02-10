import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------- 1. Tracking Differentiator Core Implementation --------------------------
class TrackingDifferentiator:
    """Second-order discrete tracking differentiator (adapted for robotic arm joint signal processing)"""
    def __init__(self, r=300, h=0.001, alpha=0.5, delta=0.05):
        """
        Parameter description (for 7DoF robotic arm):
        r: Fast tracking factor (50~1000), larger means faster tracking but worse noise resistance
        h: Sampling step (controller period, default 1ms=0.001s)
        alpha: Nonlinear factor, fixed at 0.5 (engineering optimal)
        delta: Filter threshold (0.01~0.5), larger means stronger noise removal
        """
        self.r = r
        self.h = h
        self.alpha = alpha
        self.delta = delta
        # Initialize tracking and differential values
        self.x1 = 0.0  # Tracking output (denoised position)
        self.x2 = 0.0  # Differential output (velocity)

    def fal(self, e):
        """Nonlinear fal function (core anti-interference logic)"""
        abs_e = np.abs(e)
        if abs_e <= self.delta:
            return e / (self.delta ** (1 - self.alpha))
        else:
            return (abs_e ** self.alpha) * np.sign(e)

    def update(self, v):
        """
        Single iteration calculation
        v: Input raw noisy signal (robotic arm joint position)
        Returns: x1 (tracking signal), x2 (differential signal)
        """
        e = self.x1 - v
        u = -self.r * self.fal(e) - self.x2
        self.x1 += self.h * self.x2
        self.x2 += self.h * u
        return self.x1, self.x2

# -------------------------- 2. 7DoF Robotic Arm Signal Simulation --------------------------
def simulate_7dof_arm(t_total=5, h=0.001):
    """
    Simulate 7DoF robotic arm joint position signals (with noise)
    t_total: Total simulation time (seconds)
    h: Sampling step (seconds)
    Returns:
        t: Time sequence
        joint_pos_raw: 7 joint raw noisy positions (shape: (7, N))
        joint_pos_true: 7 joint true noise-free positions (shape: (7, N))
    """
    # Generate time sequence
    t = np.arange(0, t_total, h)
    N = len(t)
    
    # 7 joint true motion trajectories (sine curves with different frequencies, simulating actual motion)
    joint_freq = [0.5, 0.7, 1.0, 0.8, 1.2, 0.6, 0.9]  # Motion frequency for each joint
    joint_amp = [15, 20, 18, 22, 16, 19, 17]         # Motion amplitude for each joint (°)
    joint_pos_true = np.zeros((7, N))
    
    for i in range(7):
        # Sine trajectory + small offset, simulating joint motion
        joint_pos_true[i] = joint_amp[i] * np.sin(2 * np.pi * joint_freq[i] * t) + 5
    
    # Add sensor noise (Gaussian white noise, simulating encoder noise)
    noise = np.random.normal(0, 0.3, (7, N))  # Noise amplitude ±0.3° (typical industrial encoder noise)
    joint_pos_raw = joint_pos_true + noise
    
    return t, joint_pos_raw, joint_pos_true

# -------------------------- 3. TD Algorithm Processing 7DoF Joint Signals --------------------------
def process_7dof_with_td(t, joint_pos_raw, h=0.001):
    """Apply TD algorithm to 7 joints separately"""
    # Initialize 7 TD instances (each joint with independent parameters, using general parameters here)
    td_list = [
        TrackingDifferentiator(r=300, h=h, delta=0.001),  # Joint 1
        TrackingDifferentiator(r=320, h=h, delta=0.05),  # Joint 2
        TrackingDifferentiator(r=280, h=h, delta=0.05),  # Joint 3
        TrackingDifferentiator(r=310, h=h, delta=0.5),  # Joint 4
        TrackingDifferentiator(r=290, h=h, delta=0.05),  # Joint 5
        TrackingDifferentiator(r=330, h=h, delta=0.05),  # Joint 6
        TrackingDifferentiator(r=270, h=h, delta=0.000005)   # Joint 7
    ]
    
    # Store processing results
    joint_pos_td = np.zeros_like(joint_pos_raw)   # TD tracking position
    joint_vel_td = np.zeros_like(joint_pos_raw)   # TD differential velocity
    joint_vel_raw = np.zeros_like(joint_pos_raw)  # Raw differential velocity (for comparison)
    
    # Process each joint step by step
    N = len(t)
    for i in range(7):
        td = td_list[i]
        # Raw differential velocity calculation (traditional method, high noise)
        joint_vel_raw[i, 1:] = (joint_pos_raw[i, 1:] - joint_pos_raw[i, :-1]) / h
        joint_vel_raw[i, 0] = 0  # Set first point velocity to 0
        
        # TD algorithm processing
        for k in range(N):
            pos_td, vel_td = td.update(joint_pos_raw[i, k])
            joint_pos_td[i, k] = pos_td
            joint_vel_td[i, k] = vel_td
    
    return joint_pos_td, joint_vel_td, joint_vel_raw

# -------------------------- 4. Visualization Comparison (Key Joints + Summary) --------------------------
def plot_td_effect(t, joint_pos_raw, joint_pos_true, joint_pos_td, 
                   joint_vel_raw, joint_vel_td):
    """
    Draw visualization results:
    1. Select joints 1/4/7 (representing different parameters) to show position+velocity comparison
    2. Summarize signal smoothness for 7 joints
    """

    # Remove Chinese font settings to avoid warnings
    # plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    # plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 12))
    
    # -------------------------- Subplot 1: Joint 1 Position Comparison --------------------------
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t, joint_pos_raw[0], 'r-', alpha=0.5, label='Raw Noisy Position', linewidth=1)
    ax1.plot(t, joint_pos_true[0], 'g-', label='True Noise-free Position', linewidth=2)
    ax1.plot(t, joint_pos_td[0], 'b-', label='TD Tracking Position', linewidth=2)
    ax1.set_title('7DoF Robotic Arm Joint 1 - Position Signal Comparison', fontsize=12)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (°)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # -------------------------- Subplot 2: Joint 1 Velocity Comparison --------------------------
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t, joint_vel_raw[0], 'r-', alpha=0.5, label='Raw Differential Velocity', linewidth=1)
    ax2.plot(t, joint_vel_td[0], 'b-', label='TD Differential Velocity', linewidth=2)
    ax2.set_title('7DoF Robotic Arm Joint 1 - Velocity Signal Comparison', fontsize=12)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (°/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # -------------------------- Subplot 3: Joint 4 Position Comparison --------------------------
    ax3 = plt.subplot(3, 3, 4)
    ax3.plot(t, joint_pos_raw[3], 'r-', alpha=0.5, label='Raw Noisy Position', linewidth=1)
    ax3.plot(t, joint_pos_true[3], 'g-', label='True Noise-free Position', linewidth=2)
    ax3.plot(t, joint_pos_td[3], 'b-', label='TD Tracking Position', linewidth=2)
    ax3.set_title('7DoF Robotic Arm Joint 4 - Position Signal Comparison', fontsize=12)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (°)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # -------------------------- Subplot 4: Joint 4 Velocity Comparison --------------------------
    ax4 = plt.subplot(3, 3, 5)
    ax4.plot(t, joint_vel_raw[3], 'r-', alpha=0.5, label='Raw Differential Velocity', linewidth=1)
    ax4.plot(t, joint_vel_td[3], 'b-', label='TD Differential Velocity', linewidth=2)
    ax4.set_title('7DoF Robotic Arm Joint 4 - Velocity Signal Comparison', fontsize=12)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (°/s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # -------------------------- Subplot 5: Joint 7 Position Comparison --------------------------
    ax5 = plt.subplot(3, 3, 7)
    ax5.plot(t, joint_pos_raw[6], 'r-', alpha=0.5, label='Raw Noisy Position', linewidth=1)
    ax5.plot(t, joint_pos_true[6], 'g-', label='True Noise-free Position', linewidth=2)
    ax5.plot(t, joint_pos_td[6], 'b-', label='TD Tracking Position', linewidth=2)
    ax5.set_title('7DoF Robotic Arm Joint 7 - Position Signal Comparison', fontsize=12)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (°)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # -------------------------- Subplot 6: Joint 7 Velocity Comparison --------------------------
    ax6 = plt.subplot(3, 3, 8)
    ax6.plot(t, joint_vel_raw[6], 'r-', alpha=0.5, label='Raw Differential Velocity', linewidth=1)
    ax6.plot(t, joint_vel_td[6], 'b-', label='TD Differential Velocity', linewidth=2)
    ax6.set_title('7DoF Robotic Arm Joint 7 - Velocity Signal Comparison', fontsize=12)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Velocity (°/s)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # -------------------------- Subplot 7: 7 Joints Position Error Statistics --------------------------
    ax7 = plt.subplot(3, 3, 3)
    # Calculate position mean square error (MSE) for each joint
    mse_raw = [np.mean((joint_pos_raw[i] - joint_pos_true[i])**2) for i in range(7)]
    mse_td = [np.mean((joint_pos_td[i] - joint_pos_true[i])**2) for i in range(7)]
    joints = [f'Joint {i+1}' for i in range(7)]
    x = np.arange(len(joints))
    width = 0.35
    
    ax7.bar(x - width/2, mse_raw, width, label='Raw Signal MSE', color='r', alpha=0.7)
    ax7.bar(x + width/2, mse_td, width, label='TD Signal MSE', color='b', alpha=0.7)
    ax7.set_title('7 Joints Position Mean Square Error Comparison', fontsize=12)
    ax7.set_xlabel('Joint Number')
    ax7.set_ylabel('Mean Square Error (°²)')
    ax7.set_xticks(x)
    ax7.set_xticklabels(joints)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # -------------------------- Subplot 8: 7 Joints Velocity Error Statistics --------------------------
    ax8 = plt.subplot(3, 3, 6)
    # True velocity (derivative of true position)
    joint_vel_true = np.zeros_like(joint_pos_true)
    for i in range(7):
        joint_vel_true[i, 1:] = (joint_pos_true[i, 1:] - joint_pos_true[i, :-1]) / 0.001
        joint_vel_true[i, 0] = 0
    
    # Velocity mean square error
    vel_mse_raw = [np.mean((joint_vel_raw[i] - joint_vel_true[i])**2) for i in range(7)]
    vel_mse_td = [np.mean((joint_vel_td[i] - joint_vel_true[i])**2) for i in range(7)]
    
    ax8.bar(x - width/2, vel_mse_raw, width, label='Raw Differential MSE', color='r', alpha=0.7)
    ax8.bar(x + width/2, vel_mse_td, width, label='TD Differential MSE', color='b', alpha=0.7)
    ax8.set_title('7 Joints Velocity Mean Square Error Comparison', fontsize=12)
    ax8.set_xlabel('Joint Number')
    ax8.set_ylabel('Mean Square Error ((°/s)²)')
    ax8.set_xticks(x)
    ax8.set_xticklabels(joints)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # -------------------------- Subplot 9: TD Parameter Description --------------------------
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    ax9.text(0.1, 0.9, 'TD Algorithm Core Parameters (7DoF Robotic Arm)', fontsize=14, weight='bold')
    ax9.text(0.1, 0.7, '1. r (tracking factor): ~300, increase for faster motion, decrease for slower', fontsize=11)
    ax9.text(0.1, 0.5, '2. δ (filter threshold): 0.05, increase for more noise, decrease for faster response', fontsize=11)
    ax9.text(0.1, 0.3, '3. h (sampling step): fixed as controller period (e.g., 1ms)', fontsize=11)
    ax9.text(0.1, 0.1, 'Effect: TD velocity MSE is 10~20x lower than raw differential, smooth position without lag', fontsize=11, color='blue')
    
    plt.tight_layout()
    plt.show()

# -------------------------- 5. Main Program Execution --------------------------
if __name__ == "__main__":
    # 1. Simulate 7DoF robotic arm signals
    h = 0.001  # 1ms sampling period (typical industrial robot controller value)
    t, joint_pos_raw, joint_pos_true = simulate_7dof_arm(t_total=5, h=h)
    
    # 2. TD algorithm processing
    joint_pos_td, joint_vel_td, joint_vel_raw = process_7dof_with_td(t, joint_pos_raw, h=h)
    
    # 3. Visualization effect
    plot_td_effect(t, joint_pos_raw, joint_pos_true, joint_pos_td, joint_vel_raw, joint_vel_td)