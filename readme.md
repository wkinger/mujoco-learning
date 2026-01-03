# 1.Configure Environment
sudo apt install libglfw3-dev
conda create --name mujoco-sim python=3.8
pip install mujoco
# 2.Tutorial
#2.1 笛卡尔空间轨迹规划
cartesian_trajectory.py
#2.2 阻抗控制
impendance_control.py

# lcm usage
cd lcm_msg 
lcm-gen -p example_t.lcm

# 阻抗控制仿真
[text](mjctrl/myopspace.py)

# issue
1. qt报警
pip uninstall pyqt5 pyqt5-tools pyqt5-Qt5 PyQt5_sip opencv-python opencv-python-headless
使用 Conda 重新安装 PyQt 和 OpenCV 包。通过 Conda 安装可以确保版本兼容性和依赖项的一致性，避免 pip 和 Conda 混用带来的冲突问题。

    conda install pyqt 
    conda install opencv
