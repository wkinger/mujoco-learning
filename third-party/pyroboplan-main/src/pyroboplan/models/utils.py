"""Utilities for model loading."""

import importlib.resources
import pyroboplan
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import subprocess
import tempfile

def get_example_models_folder():
    """
    Returns the full path the example models folder.

    Returns
    -------
        str
            The path to the `pyroboplan` example models folder.
    """
    resource_path = importlib.resources.files(pyroboplan) / "models"
    return resource_path.as_posix()



class DualArmVisualizer:
    def __init__(self, xacro_path, mesh_dir="."):
        """
        Dual Arm Robot Visualizer for xacro files
        :param xacro_path: Robot xacro file path
        :param mesh_dir: STL model file directory
        """
        # 1. Preprocess xacro file using xacro command
        urdf_path = self.preprocess_xacro_with_fixes(xacro_path)
        
        # 2. Load the preprocessed URDF file
        print(f"🔧 Loading preprocessed URDF file: {urdf_path}")
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path, package_dirs=[mesh_dir])
        
        # 3. Auto-detect and group all controllable joints
        self.joint_groups = {
            "Left Arm": ["Left_Joint1", "Left_Joint2", "Left_Joint3", "Left_Joint4", 
                        "Left_Joint5", "Left_Joint6", "Left_Joint7"],
            "Right Arm": ["Right_Joint1", "Right_Joint2", "Right_Joint3", "Right_Joint4",
                         "Right_Joint5", "Right_Joint6", "Right_Joint7"],
            "Legs": ["Knee_Joint", "Ankle_Joint"],
            "Waist": ["Waist_Pitch_Joint", "Waist_Yaw_Joint"],
            "Head": ["Head_Yaw_Joint", "Head_Pitch_Joint"],
            "Other": []  # Catch-all for joints not in other groups
        }
        
        # 4. Collect ALL controllable joints (excluding fixed joints)
        self.control_joint_ids = []
        self.control_joint_names = []
        self.joint_limits = []
        self.joint_group_map = {}  # Maps joint index to group name
        
        for jid in range(1, self.model.njoints):  # Skip root joint (ID=0)
            jname = self.model.names[jid]
            
            # Skip fixed joints (they have 0 degrees of freedom)
            if self.model.joints[jid].nv == 0:
                continue
            
            # Find which group this joint belongs to
            group_name = "Other"  # Default group for unclassified joints
            for group, prefixes in self.joint_groups.items():
                if group == "Other":
                    continue  # Skip the Other group for matching
                if any(prefix in jname for prefix in prefixes):
                    group_name = group
                    break
            
            # Add ALL controllable joints to control list
            self.control_joint_ids.append(jid)
            self.control_joint_names.append(jname)
            
            # Get joint limits from Pinocchio model
            lower = -3.14  # Default lower limit
            upper = 3.14   # Default upper limit
            
            # Try to get limits from Pinocchio model
            try:
                if hasattr(self.model, 'lowerPositionLimit') and hasattr(self.model, 'upperPositionLimit'):
                    # Get configuration index for this joint
                    config_idx = self.model.joints[jid].idx_q
                    if config_idx >= 0:
                        lower = self.model.lowerPositionLimit[config_idx]
                        upper = self.model.upperPositionLimit[config_idx]
                        print(f"  Using model limits for {jname}: [{lower:.4f}, {upper:.4f}] rad")
                    else:
                        print(f"  Joint {jname} has no configuration index, using default limits")
                else:
                    print(f"  Model has no position limits, using default limits for {jname}")
            except (IndexError, AttributeError) as e:
                print(f"  Error getting limits for {jname}: {e}, using default limits")
            
            self.joint_limits.append((lower, upper))
            self.joint_group_map[len(self.control_joint_ids) - 1] = group_name
        
        # Update Other group to include actual joint names
        other_joints = [jname for i, jname in enumerate(self.control_joint_names) 
                       if self.joint_group_map[i] == "Other"]
        self.joint_groups["Other"] = other_joints
        
        # Debug: Print all detected joints
        print(f"🤖 Detected {len(self.control_joint_ids)} controllable joints:")
        for i, jid in enumerate(self.control_joint_ids):
            jname = self.control_joint_names[i]
            group_name = self.joint_group_map[i]
            print(f"  {jname} (ID: {jid}, Group: {group_name})")
        
        # 5. Initialize joint configuration with proper index mapping
        self.q = pin.neutral(self.model)
        
        # Debug: Print model information
        print(f"🤖 Model info: nq={self.model.nq}, nv={self.model.nv}, njoints={self.model.njoints}")
        print(f"🔧 Control joint IDs: {self.control_joint_ids}")
        
        # Use proper joint configuration indices
        for i, jid in enumerate(self.control_joint_ids):
            # Get the proper index in configuration vector
            # Pinocchio uses joint.idx_q for configuration index
            config_idx = self.model.joints[jid].idx_q
            if config_idx >= 0:  # Only if joint has configuration
                # Use zero as initial value
                self.q[config_idx] = 0.0
                print(f"  Joint {jid} ({self.control_joint_names[i]}) -> config index {config_idx}, initialized to 0.0")
            else:
                print(f"⚠️  Joint {jid} has no configuration index (fixed joint?)")
        
        # Print initial configuration for verification
        print(f"🔍 Initial configuration: {self.q}")
        
        # 6. Initialize visualizer
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        self.viz.display(self.q)
        
        # 7. Create GUI interface
        self.create_gui()
    
    def preprocess_xacro_with_fixes(self, xacro_path):
        """Preprocess xacro file with proper fixes"""
        print(f"🔧 Preprocessing xacro file: {xacro_path}")
        
        # First, check if xacro command is available
        xacro_cmd = self.find_xacro_command()
        
        # Create temporary file for processed URDF
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            urdf_path = f.name
        
        try:
            # Run xacro command
            cmd = xacro_cmd + [xacro_path]
            print(f"🔧 Running command: {' '.join(cmd)}")
            
            with open(urdf_path, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                print(f"⚠️  Xacro command failed: {result.stderr}")
                # Try to manually fix the URDF
                urdf_path = self.manually_fix_urdf(urdf_path)
            else:
                print(f"✅ Xacro file processed to: {urdf_path}")
                
        except Exception as e:
            print(f"⚠️  Xacro preprocessing failed: {e}")
            # Fallback to manual processing
            urdf_path = self.manually_process_xacro(xacro_path)
        
        return urdf_path
    
    def find_xacro_command(self):
        """Find the correct xacro command"""
        # Try different xacro command variations
        commands_to_try = [
            ['xacro'],
            ['rosrun', 'xacro', 'xacro'],
            ['python3', '-m', 'xacro'],
            ['python', '-m', 'xacro']
        ]
        
        for cmd in commands_to_try:
            try:
                result = subprocess.run(cmd + ['--help'], capture_output=True)
                if result.returncode == 0:
                    print(f"✅ Found xacro command: {' '.join(cmd)}")
                    return cmd
            except:
                continue
        
        print("⚠️  No xacro command found, using manual processing")
        return ['echo']  # Fallback command
    
    def manually_process_xacro(self, xacro_path):
        """Manually process xacro file when command is not available"""
        print("🔧 Manually processing xacro file...")
        
        # Read the xacro content
        with open(xacro_path, 'r') as f:
            content = f.read()
        
        # Simple xacro processing: remove xacro tags and includes
        # This is a basic implementation for simple xacro files
        processed_content = content
        
        # Remove xacro namespace declaration
        processed_content = processed_content.replace('xmlns:xacro="http://www.ros.org/wiki/xacro"', '')
        
        # Handle xacro includes by reading and inserting included files
        import re
        include_pattern = r'<xacro:include filename="([^"]+)"\s*/>'
        
        def include_replacer(match):
            include_path = match.group(1)
            if os.path.exists(include_path):
                with open(include_path, 'r') as f:
                    included_content = f.read()
                # Remove robot tag from included content
                included_content = re.sub(r'<robot[^>]*>', '', included_content)
                included_content = re.sub(r'</robot>', '', included_content)
                return included_content
            else:
                print(f"⚠️  Included file not found: {include_path}")
                return ''
        
        processed_content = re.sub(include_pattern, include_replacer, processed_content)
        
        # Remove other xacro tags
        processed_content = re.sub(r'<xacro:[^>]*>', '', processed_content)
        processed_content = re.sub(r'</xacro:[^>]*>', '', processed_content)
        
        # Fix base_link geometry
        processed_content = processed_content.replace(
            '<link name="base_link">\n    <visual>\n      <origin xyz="0 0 0" rpy="0 0 0" />\n    </visual>\n  </link>',
            '''<link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="">
        <color rgba="0.8 0.8 0.8 1"/>
      </material>
    </visual>
  </link>'''
        )
        
        # Create temporary file for processed URDF
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            urdf_path = f.name
            f.write(processed_content)
        
        print(f"✅ Manually processed URDF created: {urdf_path}")
        return urdf_path
    
    def manually_fix_urdf(self, urdf_path):
        """Manually fix issues in the generated URDF"""
        print("🔧 Manually fixing URDF file...")
        
        # Read the generated URDF content
        with open(urdf_path, 'r') as f:
            content = f.read()
        
        # Fix common issues
        fixed_content = content
        
        # Ensure proper robot tag
        if not fixed_content.startswith('<?xml'):
            fixed_content = '<?xml version="1.0" encoding="utf-8"?>\n' + fixed_content
        
        if '<robot' not in fixed_content:
            fixed_content = fixed_content.replace('<?xml version="1.0" encoding="utf-8"?>', 
                                                '<?xml version="1.0" encoding="utf-8"?>\n<robot name="whole_robot">') + '\n</robot>'
        
        # Write fixed content back
        with open(urdf_path, 'w') as f:
            f.write(fixed_content)
        
        print(f"✅ Manually fixed URDF: {urdf_path}")
        return urdf_path
    
    def create_gui(self):
        """Create GUI slider control interface for dual arm robot"""
        # Set up figure
        plt.rcParams['toolbar'] = 'None'  # Hide toolbar
        
        # Calculate optimal figure size
        num_joints = len(self.control_joint_ids)
        max_fig_height = 15  # Maximum figure height
        fig_height = min(max_fig_height, max(8, 0.3 * num_joints))
        
        self.fig = plt.figure(figsize=(16, fig_height))
        self.fig.suptitle(f'🤖 Dual Arm Robot Control Panel - {num_joints} Controllable Joints', 
                         fontsize=16, fontweight='bold')
        
        # Create sliders for each joint
        self.sliders = []
        current_y = 0.95
        slider_height = 0.04  # More compact sliders
        group_spacing = 0.02
        
        # Group joints by category
        grouped_joints = {}
        for i, jname in enumerate(self.control_joint_names):
            group_name = self.joint_group_map[i]
            if group_name not in grouped_joints:
                grouped_joints[group_name] = []
            grouped_joints[group_name].append((i, jname))
        
        # Sort groups by importance
        group_order = ["Left Arm", "Right Arm", "Legs", "Waist", "Head", "Other"]
        sorted_groups = []
        for group in group_order:
            if group in grouped_joints and grouped_joints[group]:
                sorted_groups.append(group)
        
        # Create sliders organized by groups
        for group_name in sorted_groups:
            joints = grouped_joints[group_name]
            
            # Add group separator and title
            group_y = current_y
            
            # Add group title with joint count
            self.fig.text(0.02, group_y + 0.005, f"{group_name} ({len(joints)} joints):", 
                         fontsize=10, fontweight='bold', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7))
            
            # Create sliders for this group
            for idx, (i, jname) in enumerate(joints):
                slider_y = group_y - (idx + 1) * slider_height
                
                # Stop if we run out of space (but continue to create all sliders)
                if slider_y < 0.02:
                    slider_y = 0.02  # Minimum y position
                
                # Create slider axis
                ax = plt.axes([0.25, slider_y, 0.65, slider_height - 0.005])
                
                # Get joint limits
                lower, upper = self.joint_limits[i]
                
                # Create compact slider with proper configuration index
                jid = self.control_joint_ids[i]
                config_idx = self.model.joints[jid].idx_q
                
                slider = Slider(
                    ax=ax,
                    label=f'{jname}',
                    valmin=lower,
                    valmax=upper,
                    valinit=self.q[config_idx],
                    valfmt='%.2f rad'
                )
                
                # Bind slider callback
                slider.on_changed(self.create_slider_callback(i))
                self.sliders.append(slider)
            
            # Update current y position for next group
            current_y = group_y - (len(joints) + 1) * slider_height - group_spacing
            
            # If we run out of vertical space, warn user
            if current_y < 0.02:
                warning_text = "⚠️ Some joints may be outside the visible area. Use window scroll if available."
                self.fig.text(0.02, 0.01, warning_text, fontsize=8, color='red',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.7))
                break
        
        # Add control buttons
        button_width = 0.12
        button_spacing = 0.015
        
        reset_ax = plt.axes([0.75, 0.02, button_width, 0.04])
        self.reset_button = Button(reset_ax, 'Reset All', 
                                  color='lightcoral', hovercolor='0.9')
        self.reset_button.on_clicked(self.reset_all_joints)
        
        zero_ax = plt.axes([0.75 + button_width + button_spacing, 0.02, button_width, 0.04])
        self.zero_button = Button(zero_ax, 'Set to Zero', 
                                 color='lightgreen', hovercolor='0.9')
        self.zero_button.on_clicked(self.set_all_to_zero)
        
        # Add status display area (compact)
        self.status_text = self.fig.text(0.02, 0.98, '', fontsize=8, 
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Add joint count info
        joint_info = f"Total Controllable Joints: {num_joints}"
        self.fig.text(0.02, 0.02, joint_info, fontsize=9, 
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.7))
        
        self.update_status()
    
    def create_slider_callback(self, joint_idx):
        """Create slider callback function"""
        def callback(val):
            jid = self.control_joint_ids[joint_idx]
            # Get the proper configuration index
            config_idx = self.model.joints[jid].idx_q
            if config_idx >= 0:
                self.q[config_idx] = val
                self.viz.display(self.q)
                self.update_status()
            else:
                print(f"⚠️  Cannot update joint {jid}: no configuration index")
        return callback
    
    def reset_all_joints(self, event=None):
        """Reset all joints to zero position"""
        for i, jid in enumerate(self.control_joint_ids):
            # Get the proper configuration index
            config_idx = self.model.joints[jid].idx_q
            if config_idx >= 0:
                self.q[config_idx] = 0.0
                self.sliders[i].set_val(0.0)
        
        self.viz.display(self.q)
        self.update_status()
        print("✅ All joints reset to zero position")
    
    def set_all_to_zero(self, event=None):
        """Set all joints to zero position"""
        for i, jid in enumerate(self.control_joint_ids):
            # Get the proper configuration index
            config_idx = self.model.joints[jid].idx_q
            if config_idx >= 0:
                self.q[config_idx] = 0.0
                self.sliders[i].set_val(0.0)
        
        self.viz.display(self.q)
        self.update_status()
        print("✅ All joints set to zero position")
    
    def update_status(self):
        """Update status display with compact information"""
        status_lines = [f"📊 Joint Status ({len(self.control_joint_ids)} joints):"]
        
        # Group joints by category for better display
        grouped_status = {}
        for i, jid in enumerate(self.control_joint_ids):
            group_name = self.joint_group_map[i]
            jname = self.control_joint_names[i]
            
            # Get the proper configuration index
            config_idx = self.model.joints[jid].idx_q
            if config_idx >= 0:
                angle_rad = self.q[config_idx]
                angle_deg = np.degrees(angle_rad)
                
                if group_name not in grouped_status:
                    grouped_status[group_name] = []
                grouped_status[group_name].append(f"{jname}: {angle_rad:.2f} rad")
        
        # Build compact status text
        for group_name, joint_status in grouped_status.items():
            status_lines.append(f"\n{group_name}:")
            # Show only first few joints per group to save space
            max_joints_per_group = 3
            if len(joint_status) > max_joints_per_group:
                status_lines.extend(joint_status[:max_joints_per_group])
                status_lines.append(f"  ... and {len(joint_status) - max_joints_per_group} more")
            else:
                status_lines.extend(joint_status)
        
        self.status_text.set_text('\n'.join(status_lines))
        plt.draw()
    
    def run_gui(self):
        """Run GUI control interface"""
        num_joints = len(self.control_joint_ids)
        print("✅ Visualization interface opened (http://localhost:7001/static/)")
        print(f"✅ Dual Arm Robot GUI control panel created with {num_joints} controllable joints")
        print("📌 Controls:")
        print("  • Drag sliders to adjust joint angles")
        print("  • 'Reset All': Set all joints to zero position")
        print("  • 'Set to Zero': Set all joints to zero position")
        print("📌 Close GUI window or press Ctrl+C to exit")
        
        # Print joint groups information
        print("\n📋 Joint Groups:")
        grouped_joints = {}
        for i, jname in enumerate(self.control_joint_names):
            group_name = self.joint_group_map[i]
            if group_name not in grouped_joints:
                grouped_joints[group_name] = []
            grouped_joints[group_name].append(jname)
        
        for group_name, joints in grouped_joints.items():
            print(f"  {group_name}: {len(joints)} joints")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n👋 Program exited")
        finally:
            plt.close('all')