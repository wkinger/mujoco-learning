"""
This example shows PyRoboPlan capabilities for path planning using
Rapidly-Exploring Random Tree (RRT) algorithm on a 7-DOF Panda robot.
"""

from pinocchio.visualize import MeshcatVisualizer
import time
import argparse

from pyroboplan.core.utils import (
    extract_cartesian_poses,
    get_random_collision_free_state,
    get_random_state,
    check_collisions_at_state,
    check_collisions_at_state_with_names,
    get_joint_limits,
)
from pyroboplan.models.bionic_arm import (
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--octree",
        action="store_true",
        help="Use octree for collision detection instead of object collisions",
    )
    args = parser.parse_args()

    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)

    # if args.octree:
    #     octree = load_point_cloud(pointcloud_path=None, voxel_resolution=0.04)
    #     add_octree_collisions(model, collision_model, visual_model, octree)
    # else:
    #     add_object_collisions(model, collision_model, visual_model)

    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    get_joint_limits(model)
   
    num_tries = 0
    with Timer():
        while True:
            input("\nPress 'Enter' to change.")
            state = get_random_state(model, padding=0)
            print(f"state {state}")
            iscollid = check_collisions_at_state_with_names(model, collision_model, state, distance_padding=0)
            num_tries += 1

            print(f"Number of tries: {num_tries} iscollid {iscollid}")

            viz.display(state)
            if num_tries > 1e3:
                break

  
