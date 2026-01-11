import sys
import time
import numpy as np
import modern_robotics as mr
sys.path.append('/home/user/workspace/yaocao/teleop20251212/oculus_reader/robot/')
from meta_quest import MetaQuest, print_vr_data
from trajectoryPlan import JointTrajectory, plot_trajectory
import scipy.spatial.transform as st
# 离线仿真轨迹规划（VR+轨迹规划+画图）

if __name__ == "__main__":
    # create IK solver
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
    eomg = 0.01
    ev = 0.001
    init_joint = np.array([9, 9, 9, 9, 9, 9, 9])
    last_joint = init_joint
    init_pose = mr.FKinSpace(M, Slist, init_joint)
    pose = init_pose[:3, :3]
    dr, dp, dyaw = init_pose.as_euler('zyx', degrees=True)
    init_pose_euler = np.array([pose[0], pose[1], pose[2], dr, dp, dyaw])
    print(f"init_pose = {init_pose}")   
    # 创建轨迹规划器实例
    dt = 50 
    N = 10
    T = []
    planner = JointTrajectory(2, dt, method="cubic")  # 4个自由度，总时间5秒
    try:
        # quest = MetaQuest()
        print("🔌 MetaQuest VR设备初始化成功！")
        # 循环读取并打印数据
        i = 0
        traj_list = []
        velocity_list = []
        target_pose = []
        while i < 50:
            # quest.update()
            # vr_pose = quest.get_right_arm_increment()
            vr_pose = np.array([0, 0, 0, 0, 0, 0])
            print(f"Right Arm Increment: {vr_pose}")

            drot = st.Rotation.from_euler('zyx', vr_pose[3:], degrees=True)
            target_pose[:3] = init_pose_euler[:3] + vr_pose[:3]
            target_pose[3:] = (drot * st.Rotation.from_euler('zyx',init_pose_euler[3:],degrees=True)).as_euler('zyx', degrees=True)
            target_R  = st.Rotation.from_euler('zyx',target_pose[3:],degrees=True).as_matrix()
            T = np.hstack((target_R, target_pose[:3].reshape(-1, 1)))
            print(f"target_pose = {target_pose} T = \n{T}")
            ik_joint = mr.IKinSpace(Slist, M, T, np.deg2rad(last_joint), eomg, ev)
            if i == 0:
                cur_p = ik_joint
                continue
            next_p = ik_joint

            cur_v = 0.5 * (next_p - last_p) / dt
            print(f"cur_v {i} = {cur_v} last_v {last_v}")
            planner.multi_trajectory(last_p, cur_p, last_v, cur_v)
            traj = planner.get_trajectory(N)
            velocity = planner.get_velocy(N)
            traj_list.append(traj)
            velocity_list.append(velocity)

            last_p = cur_p
            last_v = cur_v
            cur_joint = ik_joint
            i += 1
            time.sleep(0.05)  # 20Hz刷新（匹配VR设备刷新率）

        # 将列表转换为numpy数组
        traj_list = np.concatenate(traj_list, axis=0)
        velocity_list = np.concatenate(velocity_list, axis=0)
        plot_trajectory(traj_list, velocity_list)



    except RuntimeError as e:
        print(f"❌ 初始化失败: {e}")
    except ImportError:
        print("❌ 错误：未找到MetaQuest类！请确保meta_quest.py在当前目录。")
    except KeyboardInterrupt:
        print("\n👋 程序已被用户终止。")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

