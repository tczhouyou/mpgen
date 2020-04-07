
import numpy as np

# configurations for the controllers

# kinematic chain groups and its joint configuration
LEFT_ARM_JOINT = [
                "ArmL1_Cla1",
                "ArmL2_Sho1",
                "ArmL3_Sho2",
                "ArmL4_Sho3",
                "ArmL5_Elb1",
                "ArmL6_Elb2",
                "ArmL7_Wri1",
                "ArmL8_Wri2"]
RIGHT_ARM_JOINT = [
                "ArmR1_Cla1",
                "ArmR2_Sho1",
                "ArmR3_Sho2",
                "ArmR4_Sho3",
                "ArmR5_Elb1",
                "ArmR6_Elb2",
                "ArmR7_Wri1",
                "ArmR8_Wri2"]
PLATFORM_JOINT = [
    "PlatForm_SlideX"
]
RIGHT_HAND_JOINT = [
    "Index R 1 Joint",
    "Index R 2 Joint",
    "Index R 3 Joint",
    "RightHandFingers",
    "RightHandThumb",
    "Middle R 1 Joint",
    "Middle R 2 Joint",
    "Middle R 3 Joint",
    "Pinky R 1 Joint",
    "Pinky R 2 Joint",
    "Pinky R 3 Joint",
    "Ring R 1 Joint",
    "Ring R 2 Joint",
    "Ring R 3 Joint",
    "Thumb R 1 Joint",
    "Thumb R 2 Joint",
]

LEFT_ARM_JOINT_CONFIG = {
                "ArmL1_Cla1": [-1.39626, 1.39626],
                "ArmL2_Sho1": [-1000, 1000],
                "ArmL3_Sho2": [-0.24, 3.1],
                "ArmL4_Sho3": [-1000, 1000],
                "ArmL5_Elb1": [0, 2.51],
                "ArmL6_Elb2": [-1000, 1000],
                "ArmL7_Wri1": [-0.64, 0.64],
                "ArmL8_Wri2": [-1.52, 1.52]
            }

RIGHT_ARM_JOINT_CONFIG = {
                "ArmR1_Cla1": [-1.39626, 1.39626],
                "ArmR2_Sho1": [-1000, 1000],
                "ArmR3_Sho2": [-0.26, 3.1],
                "ArmR4_Sho3": [-1000, 1000],
                "ArmR5_Elb1": [0, 2.5],
                "ArmR6_Elb2": [-1000, 1000],
                "ArmR7_Wri1": [-0.64, 0.64],
                "ArmR8_Wri2": [-1.52, 1.52]
            }

Left_HAND_JOINT_CONFIG = {
    "Index L 1 Joint": [0, 1.5708],
    "Index L 2 Joint": [0, 1.5708],
    "Index L 3 Joint": [0, 1.5708],
    "LeftHandFingers": [0, 1],
    "LeftHandThumb": [0, 1],
    "Middle L 1 Joint": [0, 1.5708],
    "Middle L 2 Joint": [0, 1.5708],
    "Middle L 3 Joint": [0, 1.5708],
    "Pinky L 1 Joint": [0, 1.5708],
    "Pinky L 2 Joint": [0, 1.5708],
    "Pinky L 3 Joint": [0, 1.5708],
    "Ring L 1 Joint": [0, 1.5708],
    "Ring L 2 Joint": [0, 1.5708],
    "Ring L 3 Joint": [0, 1.5708],
    "Thumb L 1 Joint": [0, 1.5708],
    "Thumb L 2 Joint": [0, 1.5708]
}

RIGHT_HAND_JOINT_CONFIG = {
    "Index R 1 Joint": [0, 1.5708],
    "Index R 2 Joint": [0, 1.5708],
    "Index R 3 Joint": [0, 1.5708],
    "RightHandFingers": [0, 1],
    "RightHandThumb": [0, 1],
    "Middle R 1 Joint": [0, 1.5708],
    "Middle R 2 Joint": [0, 1.5708],
    "Middle R 3 Joint": [0, 1.5708],
    "Pinky R 1 Joint": [0, 1.5708],
    "Pinky R 2 Joint": [0, 1.5708],
    "Pinky R 3 Joint": [0, 1.5708],
    "Ring R 1 Joint": [0, 1.5708],
    "Ring R 2 Joint": [0, 1.5708],
    "Ring R 3 Joint": [0, 1.5708],
    "Thumb R 1 Joint": [0, 1.5708],
    "Thumb R 2 Joint": [0, 1.5708],
}

# default controller configurations
DEFAULT_PD_CONFIG = {
    'kpos': [4500, 4500, 4500],
    'kori': [4500, 4500, 4500],
    'dpos': [20, 20, 20],
    'dori': [20, 20, 20]
}

DEFAULT_EXPLICIT_FORCE_IMPEDANCE_CONFIG = {
    'kpos': [2500, 2500, 2500],
    'kori': [1500, 1500, 1500],
    'dpos': [5, 5, 5],
    'dori': [1, 1, 1],
    'root_name': 'root~Dummy_Platform~X_Platform~Y_Platform~Yaw_Platform~Platform-colmodel',
    'u_task_filter_coeff': 1,
    'init_arm_joint_angle': [-0.068, 0.69, 0.92, -0.42, 1.5, -2.1, 0.3, 0.36],
    # 'init_arm_joint_angle': [-0.12, 0.78, 0.66, -0.55, 1.1, -1.5, -0.25, 0.25],
    'init_hand_joint_angle': np.ones(16) * 0,
    'platform_joint_list': PLATFORM_JOINT,
    'arm_name': 'RightArm',
    'arm_joint_name_list': RIGHT_ARM_JOINT,
    'arm_joint_config': RIGHT_ARM_JOINT_CONFIG,
    'hand_name': 'RightHand',
    'hand_joint_list': RIGHT_HAND_JOINT,
    'hand_joint_config': RIGHT_HAND_JOINT_CONFIG,
    'tcp_name': 'Hand R TCP',
    'ft_filter_coeff': 0.6,
    'vel_filter_coeff': 0.9,
    'enable_write_csv': False,
    # force controller configuration
    'force_kp': 0.6,  # 5.0  # 15.0
    'force_ki': 0.6,  # 1.0  # 8.0
    'force_kd': 0.001,
    # 'force_kp': 0.,
    # 'force_ki': 0.,
    # 'force_kd': 0.,
    'force_pid_anti_windup': 10,
    # 'torque_kp': 0.07,
    # 'torque_ki': 0.3,
    # 'torque_kd': 0.001,
    'torque_kp': 0.5,
    'torque_ki': 0.3,
    'torque_kd': 0.001,
    'torque_pid_anti_windup': 2,
    'contact_force_target': 10,
    'contact_torque_target': 0,
    'loose_contact_ratio': 0.2,
    'force_mag': 5,
    'target_vel_in_tool': np.zeros(6),
    'orig_tool_dir': np.array([0, 0, 1.0]),
    'rot_kp': 20,
    'rot_ki': 0.3,
    'rot_kd': 0.001,
    'rot_pid_anti_windup': 10,
    # lose contact recover
    'lcr_kp': 5,  # 10,
    'lcr_ki': 10,  # 5,
    'lcr_kd': 0.0,
    'lcr_pid_anti_windup': 200,

    'platform_kp': 0.03,
    'platform_ki': 0.0,
    'platform_kd': 0.0,
    'platform_pid_anti_windup': 2,
    'platform_vel': np.array([-0.0, 0.0, 0.0]),

    'hand_comfortable_radius': 0.2,

    'min_react_force': 3,
    'lower_angle': np.pi / 6.2,  # 6.0,
    'upper_angle': np.pi / 2.0,
    'friction_estimation_window': 500,
    'safe_friction_cone_llim': 0.2,

    'force_profile_mode': 0,  # 0: constant force target, 1: time-varying force
    'vmp_velocity_mode': True,

    # passivity observer and controller
    'passive_control_enabled': False,
    'time_window': 10,

    # anomally detection
    'max_force_limit': 100.0
}

DEFAULT_VEL_BASED_EXPLICIT_FORCE_IMPEDANCE_CONFIG = {
    'kpos': [2500, 2500, 2500],
    'kori': [1500, 1500, 1500],
    'dpos': [5, 5, 5],
    'dori': [1, 1, 1],
    # top grasp of sponge
    'init_arm_joint_angle': [-0.068, 0.69, 0.92, -0.42, 1.5, -2.1, 0.3, 0.36],
    # top right grasp of sponge
    # 'init_arm_joint_angle': [-0.12, 0.78, 0.66, -0.55, 1.1, -1.5, -0.25, 0.25],
    'init_hand_joint_angle': np.ones(16) * 0,
    'arm_name': 'RightArm',
    'arm_joint_name_list': RIGHT_ARM_JOINT,
    'arm_joint_config': RIGHT_ARM_JOINT_CONFIG,
    'hand_name': 'RightHand',
    'hand_joint_list': RIGHT_HAND_JOINT,
    'hand_joint_config': RIGHT_HAND_JOINT_CONFIG,
    'tcp_name': 'Hand R TCP',
    'ft_filter_coeff': 0.9,
    'vel_filter_coeff': 0.9,
    'enable_write_csv': False,
    # force controller configuration
    'force_kp': 0.002,  # 5.0  # 15.0
    'force_ki': 0.006,  # 1.0  # 8.0
    'force_kd': 0.0,
    # 'force_kp': 0.,
    # 'force_ki': 0.,
    # 'force_kd': 0.,
    'force_pid_anti_windup': 10,
    'torque_kp': 0.05,
    'torque_ki': 0.3,
    'torque_kd': 0.0,
    'torque_pid_anti_windup': 2,
    'contact_force_target': 10,
    'contact_torque_target': 0,
    'loose_contact_ratio': 0.2,
    'force_mag': 5,
    'target_vel_in_tool': np.zeros(6),
    'orig_tool_dir': np.array([0, 0, 1.0]),
    'rot_kp': 0.02,
    'rot_ki': 0.0,
    'rot_kd': 0.0,
    'rot_pid_anti_windup': 10,
    'min_react_force': 3,
    'lower_angle': np.pi / 6.2,  # 6.0,
    'upper_angle': np.pi / 2.0,
    'friction_estimation_window': 500,
    'safe_friction_cone_llim': 0.2,

    # passivity-based controller configuration
    'force_profile_mode': 0,  # 0: constant force target, 1: time-varying force
    'vmp_velocity_mode': True,

    # passivity observer and controller
    'passive_control_enabled': False,
    'time_window': 10,
}