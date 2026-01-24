import numpy as np
import numpy as np
def xmat_xpos_to_pose_matrix(xmat, xpos): 
    """ 将MuJoCo的xmat和xpos转换为4×4位姿矩阵

    参数:
        xmat: MuJoCo的9元素旋转矩阵（行优先展平）
        xpos: 3维位置向量 [x, y, z]
        
    返回:
        pose_matrix: 4×4齐次变换矩阵
    """
    # 创建4×4矩阵
    pose_matrix = np.eye(4)
    # 将9元素数组转换为3×3旋转矩阵
    rotation_matrix = xmat.reshape(3, 3)
    # 设置旋转部分
    pose_matrix[:3, :3] = rotation_matrix
    # 设置平移部分
    pose_matrix[:3, 3] = xpos
    
    return pose_matrix

def mju_mat2Quat(quat_out, rot_mat):
    """
    手动实现旋转矩阵转四元数（MuJoCo的mju_mat2Quat）
    :param quat_out: 输出四元数 [w, x, y, z]（预分配数组）
    :param rot_mat: 输入3×3旋转矩阵
    """
    # 确保输入是3x3矩阵
    if rot_mat.shape != (3, 3):
        # 如果是9元素数组，重塑为3x3
        if rot_mat.size == 9:
            rot_mat = rot_mat.reshape(3, 3)
        else:
            raise ValueError(f"输入矩阵形状应为(3,3)或9元素数组，但实际为{rot_mat.shape}")
    # 提取旋转矩阵元素
    r00, r01, r02 = rot_mat[0]
    r10, r11, r12 = rot_mat[1]
    r20, r21, r22 = rot_mat[2]
    
    # 计算四元数实部w
    trace = r00 + r11 + r22  # 旋转矩阵迹
    if trace > 1e-8:  # 避免w=0
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r21 - r12) * s
        y = (r02 - r20) * s
        z = (r10 - r01) * s
    else:
        # 处理trace≤0的情况（选最大对角元）
        if r00 > r11 and r00 > r22:
            s = 2.0 * np.sqrt(1.0 + r00 - r11 - r22)
            w = (r21 - r12) / s
            x = 0.25 * s
            y = (r01 + r10) / s
            z = (r02 + r20) / s
        elif r11 > r22:
            s = 2.0 * np.sqrt(1.0 + r11 - r00 - r22)
            w = (r02 - r20) / s
            x = (r01 + r10) / s
            y = 0.25 * s
            z = (r12 + r21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r22 - r00 - r11)
            w = (r10 - r01) / s
            x = (r02 + r20) / s
            y = (r12 + r21) / s
            z = 0.25 * s
    
    # 赋值到输出数组（保证单位四元数）
    quat = np.array([w, x, y, z])
    quat /= np.linalg.norm(quat)  # 归一化
    quat_out[:] = quat

def mju_negQuat(quat_conj_out: np.ndarray, quat_in: np.ndarray):
    """
    手动实现四元数共轭（MuJoCo的mju_negQuat）
    :param quat_conj_out: 输出共轭四元数 [w, -x, -y, -z]
    :param quat_in: 输入四元数 [w, x, y, z]
    """
    # 实部不变，虚部取反
    quat_conj_out[0] = quat_in[0]    # w
    quat_conj_out[1] = -quat_in[1]   # -x
    quat_conj_out[2] = -quat_in[2]   # -y
    quat_conj_out[3] = -quat_in[3]   # -z

def mju_quat2Vel(vel_out: np.ndarray, quat_error: np.ndarray, scale: float):
    """
    手动实现四元数误差转角速度（MuJoCo的mju_quat2Vel）
    :param vel_out: 输出角速度 [wx, wy, wz]
    :param quat_error: 输入四元数误差 [w, x, y, z]
    :param scale: 缩放因子（代码中为1.0）
    """
    w, x, y, z = quat_error
    
    # 更精确的实现：使用反正切函数来处理大角度旋转
    # 当实部接近±1时，使用不同的公式
    if abs(w) > 0.9999:
        # 小角度近似：角速度 ≈ 2 * 虚部
        wx = 2.0 * x * scale
        wy = 2.0 * y * scale
        wz = 2.0 * z * scale
    else:
        # 大角度情况：使用更精确的公式
        # 角速度 = 2 * arctan2(|虚部|, |实部|) * (虚部/|虚部|)
        imag_norm = np.sqrt(x*x + y*y + z*z)
        if imag_norm < 1e-8:
            # 无旋转的情况
            wx = wy = wz = 0.0
        else:
            angle = 2.0 * np.arctan2(imag_norm, abs(w))
            if w < 0:
                angle = -angle  # 处理实部为负的情况
            wx = angle * x / imag_norm * scale
            wy = angle * y / imag_norm * scale
            wz = angle * z / imag_norm * scale
    
    # 赋值到输出数组
    vel_out[0] = wx
    vel_out[1] = wy
    vel_out[2] = wz

def mju_mulQuat(quat_out: np.ndarray, quat_a: np.ndarray, quat_b: np.ndarray):
    """
    手动实现四元数乘法（哈密顿乘积，MuJoCo的mju_mulQuat）
    :param quat_out: 输出乘积四元数 q_a ⊗ q_b
    :param quat_a: 第一个四元数（目标四元数）
    :param quat_b: 第二个四元数（当前四元数的共轭）
    """
    # 提取四元数分量
    w1, x1, y1, z1 = quat_a
    w2, x2, y2, z2 = quat_b
    
    # 哈密顿乘积计算
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # 赋值到输出数组（归一化保证单位四元数）
    quat = np.array([w, x, y, z])
    quat /= np.linalg.norm(quat)
    quat_out[:] = quat





if __name__ == "__main__":
    # 测试示例
    rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 单位矩阵
    site_quat = np.zeros(4)
    mju_mat2Quat(site_quat, rot_mat)
    print("旋转矩阵转四元数结果：", site_quat)  # 输出 [1,0,0,0]（单位四元数）
    # 测试示例
    site_quat = np.array([1, 0.1, 0.2, 0.3])
    site_quat_conj = np.zeros(4)
    mju_negQuat(site_quat_conj, site_quat)
    print("四元数共轭结果：", site_quat_conj)  # 输出 [1, -0.1, -0.2, -0.3]
    # 测试示例
    tar_rot_quat = np.array([1, 0, 0, 0])  # 目标四元数（无旋转）
    site_quat_conj = np.array([1, -0.1, -0.2, -0.3])  # 当前共轭四元数
    error_quat = np.zeros(4)
    mju_mulQuat(error_quat, tar_rot_quat, site_quat_conj)
    print("四元数乘法结果：", error_quat)  # 输出 [1, -0.1, -0.2, -0.3]

    # 测试示例
    error_quat = np.array([0.9998, 0.01, 0, 0])  # 小角度误差（绕x轴旋转≈1°）
    twist_rot = np.zeros(3)
    mju_quat2Vel(twist_rot, error_quat, 1.0)
    print("四元数转角速度结果：", twist_rot)  # 输出 ≈[0.02, 0, 0]（对应角速度≈0.02 rad/s）