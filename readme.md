# 内容介绍
机器人学 + mujoco仿真环境相关的一些开源代码，在他们的基础上进行学习和实验。
目的是通过mujoco的仿真环境，对机器人学的知识进行系统学习和了解

# 目录结构
third-party:包含的第三方库。
scirpt：自己实现的一些python脚本
env：包含的仿真环境文件
util：包含的工具文件

# 安装配置
sudo apt install libglfw3-dev
conda create --name mujoco-sim python=3.8
pip install mujoco

# 包含的开源库
mujoco_menagerie-main：mujoco机器人模型和仿真环境文件
mjctrl：包含机械臂示教拖动实现、微分逆解的算法实现
third-party/ModernRobotics-master：现代机器人学代码
mujoco-learning：主要介绍mujoco仿真环境的使用方法，包含比较全面

# 自己实现的
mjctrl/myopspace.py：机械臂示教拖动
script/Otg：测试ruckig，robotoy和三次多项式轨迹生成。参考知乎[https://zhuanlan.zhihu.com/p/1986486472610690953]
script：根据知乎木一鲸家的程序员教程自己的实现 [https://zhuanlan.zhihu.com/p/705884344]


# 使用方法
#2.1 笛卡尔空间轨迹规划
cartesian_trajectory.py
#2.2 阻抗控制
impendance_control.py

# lcm usage
cd lcm_msg 
lcm-gen -p example_t.lcm

# 现存问题
1. qt报警
pip uninstall pyqt5 pyqt5-tools pyqt5-Qt5 PyQt5_sip opencv-python opencv-python-headless
使用 Conda 重新安装 PyQt 和 OpenCV 包。通过 Conda 安装可以确保版本兼容性和依赖项的一致性，避免 pip 和 Conda 混用带来的冲突问题。

    conda install pyqt 
    conda install opencv

# 下一步计划
1.整理项目结构，优化代码
2.规划规划实现mujoco仿真，观察效果。
3.机器人动力学、本体辨识等算法仿真实现。