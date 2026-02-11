"""
This example shows PyRoboPlan capabilities for path planning using
Rapidly-Exploring Random Tree (RRT) algorithm on a 7-DOF Panda robot.
"""

from pinocchio.visualize import MeshcatVisualizer
import time
import argparse
import numpy as np
import pinocchio
import meshcat
import meshcat.geometry as mg

from pyroboplan.core.utils import (
    extract_cartesian_poses,
    get_random_collision_free_state,
    get_random_state,
    check_collisions_at_state,
    check_collisions_at_state_with_names,
    get_joint_limits,
)
from pyroboplan.models.dual_arm import (
    load_models,
    load_point_cloud,
    add_self_collisions,
    add_octree_collisions,
    add_object_collisions,
)
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.visualization.meshcat_utils import visualize_frames
import sys
sys.path.append("/home/kuanwang/workspace/mujoco_ws/util/")
from timer import Timer
sys.path.append("/home/kuanwang/workspace/model_files/script/")
from read_xml import read_actuator_joint_limits, print_limits


class Logger:
    """日志输出控制类 - 高性能优化版本"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def info(self, message):
        """信息级别日志 - 避免不必要的字符串构造"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def debug(self, message):
        """调试级别日志 - 避免不必要的字符串构造"""
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def warning(self, message):
        """警告级别日志 - 避免不必要的字符串构造"""
        if self.verbose:
            print(f"[WARNING] {message}")
    
    def error(self, message):
        """错误级别日志 - 避免不必要的字符串构造"""
        if self.verbose:
            print(f"[ERROR] {message}")


class DualArmVisualizer:
    """双足机器人可视化控制器 - 高性能优化版本"""
    
    def __init__(self, model, collision_model, visual_model, data, enable_visualization=True, logger=None):
        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model
        self.data = data
        self.enable_visualization = enable_visualization
        self.logger = logger or Logger(verbose=True)
        
        # 仅在启用可视化时维护碰撞连杆记录
        if self.enable_visualization:
            self.last_collision_links = set()
        else:
            self.last_collision_links = None
        
        # 初始化可视化器 - 仅在启用可视化时执行
        if self.enable_visualization:
            self.viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
            if self.logger.verbose:
                self.logger.info("可视化器初始化完成")
        else:
            self.viz = None
            # 避免不必要的日志输出
            if self.logger.verbose:
                self.logger.info("可视化功能已禁用")
    
    def get_link_position(self, link_name, q):
        """获取连杆在世界坐标系中的位置 - 仅在启用可视化时执行"""
        if not self.enable_visualization:
            return None
            
        try:
            # 计算正运动学
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)
            
            # 查找连杆对应的frame
            for frame_id in range(self.model.nframes):
                frame = self.model.frames[frame_id]
                if frame.name == link_name:
                    # 获取连杆在世界坐标系中的变换矩阵
                    transform = self.data.oMf[frame_id]
                    # 返回位置（变换矩阵的平移部分）
                    return transform.translation
            return None
        except Exception as e:
            if self.logger.verbose:
                self.logger.error(f"获取连杆 {link_name} 位置时出错: {e}")
            return None
    
    def highlight_collision_links_simple(self, collision_links, q):
        """使用简单方法高亮显示碰撞的连杆 - 仅在启用可视化时执行"""
        if not self.enable_visualization:
            return
            
        try:
            # 清除之前的高亮显示
            try:
                self.viz.viewer["collision_highlight"].delete()
            except:
                pass
            
            # 为每个碰撞的连杆添加红色高亮
            for link_name in collision_links:
                # 获取连杆在世界坐标系中的位置
                link_position = self.get_link_position(link_name, q)
                
                if link_position is not None:
                    # 在连杆位置添加一个红色的球体作为高亮标记
                    highlight_geom = mg.Sphere(0.05)  # 小半径球体
                    highlight_material = mg.MeshBasicMaterial(
                        color=0xFF0000,  # 红色
                        transparent=True,
                        opacity=0.8
                    )
                    self.viz.viewer[f"collision_highlight/{link_name}"].set_object(highlight_geom, highlight_material)
                    
                    # 设置球体位置到连杆位置
                    transform = np.eye(4)
                    transform[:3, 3] = link_position
                    self.viz.viewer[f"collision_highlight/{link_name}"].set_transform(transform)
                    
                    if self.logger.verbose:
                        self.logger.debug(f"连杆 {link_name} 已添加红色高亮球体 (位置: {link_position.round(3)})")
                else:
                    if self.logger.verbose:
                        self.logger.warning(f"无法找到连杆 {link_name} 的位置")
                    
        except Exception as e:
            if self.logger.verbose:
                self.logger.error(f"高亮显示碰撞连杆时出错: {e}")
    
    def reset_link_colors(self):
        """重置所有连杆颜色为默认颜色 - 仅在启用可视化时执行"""
        if not self.enable_visualization:
            return
            
        try:
            for geom_obj in self.visual_model.geometryObjects:
                geom_obj.meshColor = np.array([0.7, 0.7, 0.7, 1.0])  # 默认灰色
            if self.logger.verbose:
                self.logger.debug("所有连杆颜色已重置为默认颜色")
        except Exception as e:
            if self.logger.verbose:
                self.logger.error(f"重置连杆颜色时出错: {e}")
    
    def set_link_color(self, link_name, color):
        """设置指定连杆的颜色 - 仅在启用可视化时执行"""
        if not self.enable_visualization:
            return False
            
        try:
            for geom_obj in self.visual_model.geometryObjects:
                if geom_obj.parentFrame < self.model.nframes:
                    frame = self.model.frames[geom_obj.parentFrame]
                    if frame.name == link_name:
                        geom_obj.meshColor = color
                        if self.logger.verbose:
                            self.logger.debug(f"连杆 {link_name} 颜色设置为 {color}")
                        return True
            if self.logger.verbose:
                self.logger.warning(f"未找到连杆 {link_name}")
            return False
        except Exception as e:
            if self.logger.verbose:
                self.logger.error(f"设置连杆 {link_name} 颜色时出错: {e}")
            return False
    
    def check_collisions_only(self, q, distance_padding=0.0):
        """仅执行碰撞检测，不进行任何可视化操作 - 高性能版本"""
        # 检查碰撞，禁用详细输出
        has_collision, collision_pairs = check_collisions_at_state_with_names(
            self.model, self.collision_model, q, distance_padding=distance_padding, verbose=False
        )
        return has_collision, collision_pairs
    
    def check_collisions_and_highlight_reset(self, q, distance_padding=0.0):
        """带颜色重置的碰撞检测和高亮显示函数 - 优化性能版本"""
        # 如果禁用可视化，使用高性能版本
        if not self.enable_visualization:
            return self.check_collisions_only(q, distance_padding)
        
        # 检查碰撞
        has_collision, collision_pairs = check_collisions_at_state_with_names(
            self.model, self.collision_model, q, distance_padding=distance_padding, verbose=self.logger.verbose
        )
        
        # 获取当前碰撞的连杆集合
        current_collision_links = set()
        if has_collision:
            for link1, link2 in collision_pairs:
                current_collision_links.add(link1)
                current_collision_links.add(link2)
        
        # 重置上一次碰撞的连杆颜色（如果与当前不同）
        if self.last_collision_links and self.last_collision_links != current_collision_links:
            for link_name in self.last_collision_links:
                if link_name not in current_collision_links:
                    self.set_link_color(link_name, np.array([0.89804, 0.91765, 0.92941, 1.0]))  # 重置为默认灰色
            
            # 重新加载模型应用颜色重置
            try:
                self.viz.loadViewerModel()
                self.viz.display(q)
            except Exception as e:
                if self.logger.verbose:
                    self.logger.error(f"重新加载模型失败: {e}")
        
        # 如果有碰撞，高亮显示碰撞的连杆
        if has_collision:
            if self.logger.verbose:
                for link1, link2 in collision_pairs:
                    self.logger.info(f"{link1} 与 {link2} 发生碰撞")
            
            # 设置当前碰撞连杆为红色
            for link_name in current_collision_links:
                self.set_link_color(link_name, np.array([1.0, 0.0, 0.0, 1.0]))  # 红色
            
            # 重新加载模型应用颜色更改
            try:
                self.viz.loadViewerModel()
                self.viz.display(q)
                if self.logger.verbose:
                    self.logger.debug("模型重新加载完成")
            except Exception as e:
                if self.logger.verbose:
                    self.logger.error(f"重新加载模型失败: {e}")
            
            # 更新上一次碰撞的连杆记录
            self.last_collision_links = current_collision_links.copy()
        
        return has_collision, collision_pairs
    
    def display(self, q):
        """显示机器人状态 - 仅在启用可视化时执行"""
        if self.enable_visualization:
            try:
                self.viz.display(q)
            except Exception as e:
                if self.logger.verbose:
                    self.logger.error(f"显示机器人状态失败: {e}")


def main():
    """主函数 - 高性能优化版本"""
    parser = argparse.ArgumentParser(description="双足机器人碰撞检测与可视化")
    parser.add_argument(
        "--octree",
        action="store_true",
        help="Use octree for collision detection instead of object collisions",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="启用可视化功能 (默认: False)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="禁用可视化功能",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="启用详细日志输出 (默认: False)",
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="禁用详细日志输出",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=1000,
        help="最大尝试次数 (默认: 1000)",
    )
    args = parser.parse_args()

    # 创建日志记录器
    logger = Logger(verbose=args.verbose)
    
    # 仅在启用日志时输出初始化信息
    if args.verbose:
        logger.info("开始初始化双足机器人模型...")
    
    # 创建模型和数据
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    
    # if args.octree:
    #     octree = load_point_cloud(pointcloud_path=None, voxel_resolution=0.04)
    #     add_octree_collisions(model, collision_model, visual_model, octree)
    # else:
    #     add_object_collisions(model, collision_model, visual_model)

    data = model.createData()
    collision_data = collision_model.createData()
    
    # 初始化可视化控制器
    visualizer = DualArmVisualizer(
        model, collision_model, visual_model, data,
        enable_visualization=args.visualize,
        logger=logger
    )
    
    # 仅在启用日志时输出配置信息
    if args.verbose:
        logger.info(f"可视化功能: {'启用' if args.visualize else '禁用'}")
        logger.info(f"日志输出: {'详细' if args.verbose else '简洁'}")
        logger.info(f"最大尝试次数: {args.max_tries}")
    
    num_tries = 0
    get_joint_limits(model)
    
    if args.verbose:
        logger.info("开始碰撞检测循环...")
    
    with Timer():
        while num_tries < args.max_tries:
            if args.visualize:
                input("\nPress 'Enter' to change.")
            # 生成随机状态
            state = get_random_state(model, padding=0)
            
            # 使用优化的碰撞检测函数
            iscollid, collision_pairs = visualizer.check_collisions_and_highlight_reset(
                state, distance_padding=0
            )
            num_tries += 1
            
            # 仅在启用详细日志时输出调试信息
            if args.verbose:
                logger.debug(f"尝试次数: {num_tries}, 碰撞状态: {iscollid}")
            
            # 显示机器人状态（仅在启用可视化时执行）
            visualizer.display(state)
            
            if num_tries >= args.max_tries:
                break
    
    # 仅在启用日志时输出完成信息
    if args.verbose:
        logger.info("碰撞检测循环完成")
        logger.info(f"总共尝试次数: {num_tries}")


if __name__ == "__main__":
    main()