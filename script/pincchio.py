import pinocchio as pin
"""
1. LoadModel的方法
  pin.buildModelFromXXX(XXX=Urdf, MJCF)
  RobotWrapper.BuildFromXXX(XXX=URDF, MJCF)
  注意SDF和SDRF文件也可以, 但是笔者未尝试过
"""
robot_model = RobotWrapper.BuildFromURDF(Conf.urdf_robot_filename, Conf.urdf_robot_directory, pin.JointModelFreeFlyer())
model = robot_model.model
visual_model = robot_model.visual_model
collision_model = robot_model.collision_model

model = pin.buildModelFromUrdf(Conf.urdf_robot_filename, pin.JointModelFreeFlyer())
