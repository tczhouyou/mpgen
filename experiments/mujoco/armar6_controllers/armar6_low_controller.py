import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')
os.sys.path.insert(0, '../..')
os.sys.path.insert(0, '../../..')

import mujoco_py
import mujoco_py.generated.const as const
import numpy as np
import datetime
from random import random
from math_tools.MathTools import *
from config.controller_configurations import *
import platform
import pandas as pd

# if platform.system() is not 'Darwin':
#     import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
from python_pid_controller.PID import PID
from sklearn.decomposition import PCA

from scipy.linalg import null_space

LEFT_JOINT = [
    "ArmL1_Cla1",
    "ArmL2_Sho1",
    "ArmL3_Sho2",
    "ArmL4_Sho3",
    "ArmL5_Elb1",
    "ArmL6_Elb2",
    "ArmL7_Wri1",
    "ArmL8_Wri2"]

RIGHT_JOINT = [
    "ArmR1_Cla1",
    "ArmR2_Sho1",
    "ArmR3_Sho2",
    "ArmR4_Sho3",
    "ArmR5_Elb1",
    "ArmR6_Elb2",
    "ArmR7_Wri1",
    "ArmR8_Wri2"]

LEFT_JOINT_CONFIG = {
    "ArmL1_Cla1": [-1.39626, 1.39626],
    "ArmL2_Sho1": [-1000, 1000],
    "ArmL3_Sho2": [-0.24, 3.1],
    "ArmL4_Sho3": [-1000, 1000],
    "ArmL5_Elb1": [0, 2.51],
    "ArmL6_Elb2": [-1000, 1000],
    "ArmL7_Wri1": [-0.64, 0.64],
    "ArmL8_Wri2": [-1.52, 1.52]
}

RIGHT_JOINT_CONFIG = {
    "ArmR1_Cla1": [-1.39626, 1.39626],
    "ArmR2_Sho1": [-1000, 1000],
    "ArmR3_Sho2": [-0.26, 3.1],
    "ArmR4_Sho3": [-1000, 1000],
    "ArmR5_Elb1": [0, 2.5],
    "ArmR6_Elb2": [-1000, 1000],
    "ArmR7_Wri1": [-0.64, 0.64],
    "ArmR8_Wri2": [-1.52, 1.52]
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
    "Thumb R 2 Joint": [0, 1.5708]
}

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
    "Thumb R 2 Joint"
]


def get_actuator_data(raw, ids_in_order):
    if raw.ndim > 1:
        return raw[:, ids_in_order]
    else:
        return raw[ids_in_order]


def get_sensor_value(sim, dim, sensorName: str):
    if sensorName not in sim.model.sensor_names:
        ret = []
        for _ in range(dim):
            ret.append(0)
        return np.array(ret)

    index = sim.model.sensor_name2id(sensorName)
    adr = sim.model.sensor_adr[index]
    ret = []
    for i in range(dim):
        ret.append(sim.data.sensordata[adr + i])
    return np.array(ret)


DEFAULT_PD_CONFIG = {
    'kpos': [4500, 4500, 4500],
    'kori': [4500, 4500, 4500],
    # 'kori': [100,100,100],
    'dpos': [20, 20, 20],
    'dori': [20, 20, 20]
}

DEFAULT_PD_VEL_CONFIG = {
    'kpos': [100, 100, 100],
    'kori': [1, 1, 1],
    # 'kori': [100,100,100],
    'dpos': [0, 0, 0],
    'dori': [0, 0, 0]
}


class HandController:
    def __init__(self, model, sim, arm_name="RightArm"):
        self.sim = sim
        if arm_name == "RightArm":
            self.joint_name_list = RIGHT_HAND_JOINT
            self.jointConfig = RIGHT_HAND_JOINT_CONFIG

        self.n_joint = len(self.joint_name_list)
        self.actuator_ids = {joint: model.actuator_name2id(joint + '_velocity') for joint in self.jointConfig}
        self.actuator_id_in_order = [self.actuator_ids[joint_name] for joint_name in self.joint_name_list]

    def close(self):
        for j, jointName in enumerate(self.joint_name_list):
            if self.sim.data.qpos[self.actuator_ids[jointName]] < self.jointConfig[jointName][1]:
                self.sim.data.ctrl[self.actuator_ids[jointName]] = 40
            else:
                self.sim.data.ctrl[self.actuator_ids[jointName]] = 0.0


    def open(self):
        for j, jointName in enumerate(self.joint_name_list):
            if self.sim.data.qpos[self.actuator_ids[jointName]] > self.jointConfig[jointName][0]:
                self.sim.data.ctrl[self.actuator_ids[jointName]] = -40
            else:
                print('here')
                self.sim.data.ctrl[self.actuator_ids[jointName]] = 0.0



class JointController:
    def __init__(self, model, sim, arm_name="RightArm", ctrl_mode="Torque"):
        self.sim = sim

        if ctrl_mode == "Position":
            ctrl_suffix = '_position'
        elif ctrl_mode == "Velocity":
            ctrl_suffix = '_velocity'
        else:
            ctrl_suffix = '_motor'

        if arm_name == "LeftArm":
            self.joint_name_list = LEFT_JOINT
            self.jointConfig = LEFT_JOINT_CONFIG
            self.tcp_name = "Hand L TCP"
        else:
            self.joint_name_list = RIGHT_JOINT
            self.jointConfig = RIGHT_JOINT_CONFIG
            self.tcp_name = "Hand R TCP"

        self.n_joint = len(self.joint_name_list)
        self.actuator_ids = {joint: model.actuator_name2id(joint + ctrl_suffix) for joint in self.jointConfig}
        self.actuator_id_in_order = [self.actuator_ids[joint_name] for joint_name in self.joint_name_list]

    def control(self, cmd):
        torque_bias = get_actuator_data(self.sim.data.qfrc_bias, self.actuator_id_in_order)
        cmd = cmd + torque_bias
        for j, jointName in enumerate(self.joint_name_list):
            self.sim.data.ctrl[self.actuator_ids[jointName]] = cmd[j]

        return cmd


class JointVelocityController:
    def __init__(self, model, sim, arm_name="RightArm"):
        self.sim = sim

        if arm_name == "LeftArm":
            self.joint_name_list = LEFT_JOINT
            self.jointConfig = LEFT_JOINT_CONFIG
            self.tcp_name = "Hand L TCP"
        else:
            self.joint_name_list = RIGHT_JOINT
            self.jointConfig = RIGHT_JOINT_CONFIG
            self.tcp_name = "Hand R TCP"

        self.n_joints = len(self.joint_name_list)
        self.actuator_ids = {joint: model.actuator_name2id(joint + "_velocity") for joint in self.jointConfig}
        self.actuator_id_in_order = [self.actuator_ids[joint_name] for joint_name in self.joint_name_list]

    def control(self, targets):
        for jname, _ in targets.items():
            self.sim.data.ctrl[self.actuator_ids[jname]] = targets[jname] * 180/ np.pi


class TaskSpaceVelocityController:
    def __init__(self, model, sim, config=None, arm_name="RightArm", desired_joints=None):
        self.sim = sim

        if config is None:
            self.config = DEFAULT_PD_VEL_CONFIG
        else:
            self.config = config

        self.kpos = self.config['kpos']
        self.kori = self.config['kori']
        self.dpos = self.config['dpos']
        self.dori = self.config['dori']

        if arm_name == "LeftArm":
            self.joint_name_list = LEFT_JOINT
            self.jointConfig = LEFT_JOINT_CONFIG
            self.tcp_name = "Hand L TCP"
        else:
            self.joint_name_list = RIGHT_JOINT
            self.jointConfig = RIGHT_JOINT_CONFIG
            self.tcp_name = "Hand R TCP"

        self.n_joints = len(self.joint_name_list)
        self.actuator_ids = {joint: model.actuator_name2id(joint + "_velocity") for joint in self.jointConfig}
        self.actuator_id_in_order = [self.actuator_ids[joint_name] for joint_name in self.joint_name_list]

        if desired_joints is None:
            self.static_nullspace_joints = np.array([0, 0, 0, 0, 1.6, 3.14, 0, 0])
        else:
            self.static_nullspace_joints = desired_joints


    def read(self, targets):

        if 'position' in targets.keys():
            target_xyz = targets['position']
        else:
            print('target is invalid, because key word {} is missing'.format('position'))

        if 'orientation' in targets.keys():
            target_quat = targets['orientation']
        else:
            print('target is invalid, because key word {} is missing'.format('orientation'))

        if 'nullspace' in targets.keys():
            desired_joints = targets['nullspace']
        else:
            desired_joints = None

        return target_xyz, target_quat, desired_joints


    def control(self, targets):
        target_xyz, target_quat, desired_joints = self.read(targets)

        tcp_xpos = self.sim.data.get_site_xpos(self.tcp_name)
        tcp_xmat = self.sim.data.get_site_xmat(self.tcp_name)
        v_p = self.sim.data.get_site_xvelp(self.tcp_name)
        v_r = self.sim.data.get_site_xvelr(self.tcp_name)
        qpos = get_actuator_data(self.sim.data.qpos, self.actuator_id_in_order)
        qvel = get_actuator_data(self.sim.data.qvel, self.actuator_id_in_order)

        # inverse kinematic control
        # position
        u_task = np.zeros(6)
        u_task[:3] = np.multiply(self.kpos, (target_xyz - tcp_xpos)) - np.multiply(self.dpos, v_p)

        # orientation
        rot_mat = np.zeros(9)
        mujoco_py.functions.mju_quat2Mat(rot_mat, target_quat)
        target_mat = rot_mat.reshape(3, 3)
        rot_diff_mat = target_mat.dot(np.linalg.inv(tcp_xmat))
        rpy_error = mat2rpy(rot_diff_mat)
        u_task[3:] = np.multiply(self.kori, rpy_error) - np.multiply(self.dori, v_r)

        # get jacobian matrix
        jac_p = self.sim.data.get_site_jacp(self.tcp_name)
        jac_r = self.sim.data.get_site_jacr(self.tcp_name)
        jac = np.zeros((6, self.n_joints))

        jac[0:3, :] = get_actuator_data(jac_p.reshape(3, -1), self.actuator_id_in_order)
        jac[3:, :] = get_actuator_data(jac_r.reshape(3, -1), self.actuator_id_in_order)

        njac = null_space(jac)

        # get velocity
        jmat = np.linalg.pinv(jac, 1e-10)
        jvel = jmat.dot(u_task)

        # null space control
        if desired_joints is None:
            u_null = 20 * (self.static_nullspace_joints - qpos) - 4 * qvel
        else:
            u_null = 20 * (desired_joints - qpos) - 4 * qvel

        nmat = njac.dot(np.linalg.pinv(njac, 1e-10))
        njvel = nmat.dot(u_null)
        jvel = jvel + njvel

        jvel = np.clip(jvel, a_min=-200, a_max=200)
        for j, jointName in enumerate(self.joint_name_list):
            self.sim.data.ctrl[self.actuator_ids[jointName]] = jvel[j] * 180 / np.pi

        return jvel


class TaskSpaceImpedanceController:
    def __init__(self, model, sim, config=None, arm_name="RightArm", desired_joints=None):
        self.sim = sim

        if config is None:
            self.config = DEFAULT_PD_CONFIG
        else:
            self.config = config

        self.kpos = self.config['kpos']
        self.kori = self.config['kori']
        self.dpos = self.config['dpos']
        self.dori = self.config['dori']

        if arm_name == "LeftArm":
            self.joint_name_list = LEFT_JOINT
            self.jointConfig = LEFT_JOINT_CONFIG
            self.tcp_name = "Hand L TCP"
        else:
            self.joint_name_list = RIGHT_JOINT
            self.jointConfig = RIGHT_JOINT_CONFIG
            self.tcp_name = "Hand R TCP"

        self.n_joints = len(self.joint_name_list)
        self.actuator_ids = {joint: model.actuator_name2id(joint + "_motor") for joint in self.jointConfig}
        self.actuator_id_in_order = [self.actuator_ids[joint_name] for joint_name in self.joint_name_list]

        if desired_joints is None:
            self.static_nullspace_joints = np.array([0, 0, 0, 0, 1.6, 3.14, 0, 0])
        else:
            self.static_nullspace_joints = desired_joints

    def read(self, targets):

        if 'position' in targets.keys():
            target_xyz = targets['position']
        else:
            print('target is invalid, because key word {} is missing'.format('position'))

        if 'orientation' in targets.keys():
            target_quat = targets['orientation']
        else:
            print('target is invalid, because key word {} is missing'.format('orientation'))

        if 'nullspace' in targets.keys():
            desired_joints = targets['nullspace']
        else:
            desired_joints = None

        return target_xyz, target_quat, desired_joints

    def control(self, targets):
        target_xyz, target_quat, desired_joints = self.read(targets)

        tcp_xpos = self.sim.data.get_site_xpos(self.tcp_name)
        tcp_xmat = self.sim.data.get_site_xmat(self.tcp_name)
        v_p = self.sim.data.get_site_xvelp(self.tcp_name)
        v_r = self.sim.data.get_site_xvelr(self.tcp_name)
        qpos = get_actuator_data(self.sim.data.qpos, self.actuator_id_in_order)
        qvel = get_actuator_data(self.sim.data.qvel, self.actuator_id_in_order)
        torque_bias = get_actuator_data(self.sim.data.qfrc_bias, self.actuator_id_in_order)

        # inverse dynamic control
        # position
        u_task = np.zeros(6)
        u_task[:3] = np.multiply(self.kpos, (target_xyz - tcp_xpos)) - np.multiply(self.dpos, v_p)

        # orientation
        rot_mat = np.zeros(9)
        mujoco_py.functions.mju_quat2Mat(rot_mat, target_quat)
        target_mat = rot_mat.reshape(3, 3)
        rot_diff_mat = target_mat.dot(np.linalg.inv(tcp_xmat))
        rpy_error = mat2rpy(rot_diff_mat)
        u_task[3:] = np.multiply(self.kori, rpy_error) - np.multiply(self.dori, v_r)

        # calculate torque
        jac_p = self.sim.data.get_site_jacp(self.tcp_name)
        jac_r = self.sim.data.get_site_jacr(self.tcp_name)
        jac = np.zeros((6, self.n_joints))

        jac[0:3, :] = get_actuator_data(jac_p.reshape(3, -1), self.actuator_id_in_order)
        jac[3:, :] = get_actuator_data(jac_r.reshape(3, -1), self.actuator_id_in_order)
        torque = jac.T.dot(u_task)

        # null space control
        if desired_joints is None:
            u_null = 20 * (self.static_nullspace_joints - qpos) - 4 * qvel
        else:
            u_null = 20 * (desired_joints - qpos) - 4 * qvel

        NullMat = np.identity(self.n_joints)
        jacT_pinv = np.linalg.pinv(jac.T, 1e-3)
        NullMat = NullMat - jac.T.dot(jacT_pinv)
        u_null = np.clip(u_null, a_min=-50, a_max=50)
        torque_null = NullMat.dot(u_null)

        torque_cmd = torque + torque_null + torque_bias
        torque_cmd = np.clip(torque_cmd, a_min=-70, a_max=70)

        # set command to Mujoco for each joint
        for j, jointName in enumerate(self.joint_name_list):
            self.sim.data.ctrl[self.actuator_ids[jointName]] = torque_cmd[j]

        return torque_cmd


class TaskSpaceExplicitForceImpedanceController:
    def __init__(self, model, sim, config=None, arm_name="RightArm", desired_joints=None):
        self.sim = sim

        if config is None:
            self.config = DEFAULT_EXPLICIT_FORCE_IMPEDANCE_CONFIG
        else:
            self.config = config

        self.restart_high_ctrl = False

        self.kpos = self.config['kpos']
        self.kori = self.config['kori']
        self.dpos = self.config['dpos']
        self.dori = self.config['dori']

        self.arm_name = self.config['arm_name']
        self.arm_joint_name_list = self.config['arm_joint_name_list']
        self.arm_joint_config = self.config['arm_joint_config']
        self.tcp_name = self.config['tcp_name']

        self.n_joints = len(self.arm_joint_name_list)
        self.actuator_ids = {joint: model.actuator_name2id(joint + "_motor") for joint in self.arm_joint_name_list}
        self.actuator_id_in_order = [self.actuator_ids[joint_name] for joint_name in self.arm_joint_name_list]
        self.init_hand_joint_angle = self.config['init_hand_joint_angle']
        self.init_arm_joint_angle = self.config['init_arm_joint_angle']
        # print("joint ids:", self.actuator_id_in_order)

        if desired_joints is None:
            self.static_nullspace_joints = self.config['init_arm_joint_angle']
        else:
            self.static_nullspace_joints = desired_joints

        self.hand_name = self.config['hand_name']
        self.hand_joint_list = self.config['hand_joint_list']
        self.hand_joint_config = self.config['hand_joint_config']
        self.tcp_name = self.config['tcp_name']
        self.ft_filter_coeff = self.config['ft_filter_coeff']
        self.vel_filter_coeff = self.config['vel_filter_coeff']
        self.enable_write_csv = self.config['enable_write_csv']

        # force controller configuration
        self.force_kp = self.config['force_kp']
        self.force_ki = self.config['force_ki']
        self.force_kd = self.config['force_kd']
        self.contact_force_target = self.config['contact_force_target']
        self.force_mag = self.config['force_mag']
        self.target_vel_in_tool = self.config['target_vel_in_tool']
        self.orig_tool_dir = self.config['orig_tool_dir']
        self.rot_kp = self.config['rot_kp']
        self.rot_ki = self.config['rot_ki']
        self.rot_kd = self.config['rot_kd']
        self.min_react_force = self.config['min_react_force']
        self.lower_angle = self.config['lower_angle']
        self.upper_angle = self.config['upper_angle']
        self.force_pid = PID(self.force_kp, self.force_ki, self.force_kd, current_time=sim.data.time)
        self.force_pid.setWindup(10)
        self.friction_estimation_window = self.config['friction_estimation_window']
        self.mu = 5.0

        self.torque_pid = PID(self.rot_kp, self.rot_ki, self.rot_kd, current_time=sim.data.time)
        self.torque_pid.setWindup(4)

        self.last_ft_force = np.zeros(3, dtype="float64")
        self.last_ft_torque = np.zeros(3, dtype="float64")
        self.last_vel = np.zeros(6, dtype="float64")

        # passivity-based controller configuration
        self.force_profile_mode = self.config['force_profile_mode']
        self.force_mag = 5
        self.vmp_velocity_mode = self.config['vmp_velocity_mode']
        self.passive_control_enabled = self.config['passive_control_enabled']
        self.time_window = self.config['time_window']

        self.passivity_observer = 0
        self.rc = 1
        self.vc = 0
        self.freez_counter = 0
        self.passive_counter = 0
        self.frozen = False

        np.set_printoptions(threshold=np.inf, edgeitems=3, linewidth=150, formatter=dict(float=lambda x: "%.2g" % x))
        self.setup_plot()
        self.set_init_pose()
        self.first_run()

    def set_init_pose(self):
        states = self.sim.get_state()
        for joint_id, joint in enumerate(self.arm_joint_name_list):
            upper_bound = self.arm_joint_config[joint][1]
            lower_bound = self.arm_joint_config[joint][0]
            if upper_bound > 3.14:
                upper_bound = 3.14
            if lower_bound < - 3.14:
                lower_bound = - 3.14
            value = random() * (upper_bound - lower_bound) + lower_bound
            addr = self.sim.model.get_joint_qpos_addr(joint)
            # states.qpos[addr] = value
            states.qpos[addr] = self.init_arm_joint_angle[joint_id]

        for joint_id, joint in enumerate(self.hand_joint_list):
            upper_bound = self.hand_joint_config[joint][1]
            lower_bound = self.hand_joint_config[joint][0]
            if upper_bound > 3.14:
                upper_bound = 3.14
            if lower_bound < - 3.14:
                lower_bound = - 3.14
            value = random() * (upper_bound - lower_bound) + lower_bound
            addr = self.sim.model.get_joint_qpos_addr(joint)
            # states.qpos[addr] = value
            states.qpos[addr] = self.init_hand_joint_angle[joint_id]

        # bottle_init = [-.6, 1.3, .95, 1, 0, 0, 0]
        # addr = self.sim.model.get_joint_qpos_addr("Jbottle")
        # for i in range(addr[1] - addr[0]):
        #     states.qpos[addr[0] + i] = bottle_init[i]

        self.sim.set_state(states)
        print("set initial state done")
        self.sim.forward()

    def setup_plot(self):
        plt.ion()
        fig = plt.figure(figsize=plt.figaspect(1))
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)
        ax1.set_title('3d position: dmp and tracking')
        ax2.set_title('force in tool frame')
        ax3.set_title('torque in tool frame')
        ax4.set_title('passive observer')
        ax5.set_title('passive observer moving mean')
        ax6.set_title('Resistance Rc')
        plt.draw()
        plt.pause(0.0005)
        self.axes = [ax1, ax2, ax3, ax4, ax5, ax6]

        self.record_motion = []
        self.record_dmp = []
        self.record_force = []
        self.record_force_root = []
        self.record_torque = []
        self.record_force_target = []
        self.record_time = []
        self.record_data = []
        self.record_angle = []
        self.record_po = []
        self.record_pobs = []
        self.record_rc = []
        self.record_velocity = []
        self.record_mu = []
        self.record_friction = []
        self.record_friction_norm = []
        self.record_norm_force = []
        self.data_current_step = np.zeros(33)

    def plot(self):
        # for plotting
        recorded_motion = np.array(self.record_motion)
        recorded_dmp = np.array(self.record_dmp)
        recorded_time = np.array(self.record_time)
        recorded_force = np.array(self.record_force)
        recorded_force_root = np.array(self.record_force_root)
        recorded_torque = np.array(self.record_torque)
        recorded_po = np.array(self.record_po)
        recorded_pos = np.array(self.record_pobs)
        recorded_rc = np.array(self.record_rc)
        recorded_angle = np.array(self.record_angle)
        recorded_force_target = np.array(self.record_force_target)
        recorded_velocity = np.array(self.record_velocity)
        recorded_mu = np.array(self.record_mu)
        recorded_friction = np.array(self.record_friction)
        recorded_friction_norm = np.array(self.record_friction_norm)
        recorded_norm_force = np.array(self.record_norm_force)
        # if print_once:
        self.axes[0].plot3D(recorded_motion[:, 0], recorded_motion[:, 1], recorded_motion[:, 2], color="orange")
        self.axes[0].plot3D(recorded_dmp[:, 0], recorded_dmp[:, 1], recorded_dmp[:, 2], color="blue")
        self.axes[1].plot(recorded_time, recorded_force[:, 0], color='red', alpha=0.4)
        self.axes[1].plot(recorded_time, recorded_force[:, 1], color='green', alpha=0.4)
        self.axes[1].plot(recorded_time, recorded_friction[:, 0], color='red', alpha=0.2, linestyle=":")
        self.axes[1].plot(recorded_time, recorded_friction[:, 1], color='green', alpha=0.2, linestyle=":")
        # self.axes[1].plot(recorded_time, recorded_force[:, 2], color='blue', alpha=0.4)
        self.axes[1].plot(recorded_time, recorded_force_target, color='orange', alpha=0.4)
        self.axes[2].plot(recorded_time, recorded_torque[:, 0], color='red', alpha=0.4)
        self.axes[2].plot(recorded_time, recorded_torque[:, 1], color='green', alpha=0.4)
        self.axes[2].plot(recorded_time, recorded_torque[:, 2], color='blue', alpha=0.4)

        self.axes[3].plot(recorded_time, recorded_po, alpha=0.4)
        # self.axes[4].plot(recorded_time, recorded_pos, alpha=0.4)
        self.axes[4].plot(recorded_time, recorded_force_root[:, 0], color='red', alpha=0.4)
        self.axes[4].plot(recorded_time, recorded_force_root[:, 1], color='green', alpha=0.4)
        self.axes[4].plot(recorded_time, recorded_force_root[:, 2], color='blue', alpha=0.4)
        # self.axes[4].plot(recorded_time, -recorded_velocity[:, 0], color='red', alpha=0.4)
        # self.axes[4].plot(recorded_time, -recorded_velocity[:, 1], color='green', alpha=0.4)
        # self.axes[4].plot(recorded_time, recorded_velocity[:, 2], color='blue', alpha=0.4)
        # self.axes[5].plot(recorded_norm_force, recorded_friction_norm, alpha=0.4)
        self.axes[5].plot(recorded_time, recorded_mu, alpha=0.4)
        # self.axes[5].plot(recorded_time, recorded_rc, alpha=0.4)
        # self.axes[5].plot(recorded_time, recorded_angle, alpha=0.4)
        self.record_time.clear()
        self.record_force.clear()
        self.record_force_root.clear()
        self.record_force_target.clear()
        self.record_torque.clear()
        self.record_po.clear()
        self.record_pobs.clear()
        self.record_rc.clear()
        self.record_angle.clear()
        self.record_velocity.clear()
        self.record_mu.clear()
        self.record_friction.clear()
        # self.record_friction_norm.clear()
        plt.draw()
        plt.pause(0.0001)

    def first_run(self):
        self.start_time = self.sim.data.time
        self.last_time = self.sim.data.time

        ft_sensor_xmat = self.sim.data.get_site_xmat("SITE_FT")
        tcp_xpos = self.sim.data.get_site_xpos("Hand R TCP")
        tcp_xmat = self.sim.data.get_site_xmat("Hand R TCP")
        self.tool_to_hand_xmat = tcp_xmat.copy().T
        # self.tool_xmat = np.identity(3)
        self.tool_dir = np.array([0, 0, 1])

        self.ft_force_init = -get_sensor_value(self.sim, 3, "Force_R")
        self.ft_torque_init = -get_sensor_value(self.sim, 3, "Torque_R")

        # self.last_tcp_xpos_target = tcp_xpos.copy()

        self.target_xpos = tcp_xpos.copy()
        self.target_xmat = tcp_xmat.copy()

        self.target_force_in_tool = np.zeros(6)

    def read(self, targets):

        if 'position' in targets.keys():
            target_xyz = targets['position']
        else:
            print('target is invalid, because key word {} is missing'.format('position'))

        if 'orientation' in targets.keys():
            target_quat = targets['orientation']
        else:
            print('target is invalid, because key word {} is missing'.format('orientation'))

        if 'nullspace' in targets.keys():
            desired_joints = targets['nullspace']
        else:
            desired_joints = None

        if 'canonical_value' in targets.keys():
            canonical_value = targets['canonical_value']
        else:
            canonical_value = None

        return target_xyz, target_quat, desired_joints, canonical_value

    def control(self, targets):
        target_xyz, target_quat, desired_joints, normalizedTime = self.read(targets)
        # print("ctl: ", target_xyz, target_quat, normalizedTime)

        # position, velocity feedback
        tcp_xpos = self.sim.data.get_site_xpos(self.tcp_name).copy()
        tcp_xmat = self.sim.data.get_site_xmat(self.tcp_name).copy()
        v_p = self.sim.data.get_site_xvelp(self.tcp_name).copy()
        v_r = self.sim.data.get_site_xvelr(self.tcp_name).copy()
        v_p = self.vel_filter_coeff * v_p + (1 - self.vel_filter_coeff) * self.last_vel[:3]
        v_r = self.vel_filter_coeff * v_r + (1 - self.vel_filter_coeff) * self.last_vel[3:]
        self.last_vel[:3] = v_p.copy()
        self.last_vel[3:] = v_r.copy()
        qpos = get_actuator_data(self.sim.data.qpos, self.actuator_id_in_order)
        qvel = get_actuator_data(self.sim.data.qvel, self.actuator_id_in_order)
        # print("qpos: ", qpos)

        # force feedback
        ft_force = -get_sensor_value(self.sim, 3, "Force_R") - self.ft_force_init
        ft_torque = -get_sensor_value(self.sim, 3, "Torque_R") - self.ft_torque_init
        ft_force = self.ft_filter_coeff * ft_force + (1 - self.ft_filter_coeff) * self.last_ft_force
        ft_torque = self.ft_filter_coeff * ft_torque + (1 - self.ft_filter_coeff) * self.last_ft_torque
        self.last_ft_force = ft_force.copy()
        self.last_ft_torque = ft_torque.copy()
        ft_sensor_xpos = self.sim.data.get_site_xpos("SITE_FT")
        ft_sensor_xmat = self.sim.data.get_site_xmat("SITE_FT")
        tool_xmat = tcp_xmat.dot(self.tool_to_hand_xmat)
        force_in_root = ft_sensor_xmat.dot(ft_force)
        torque_in_root = ft_sensor_xmat.dot(ft_torque)
        force_in_tool = tool_xmat.T.dot(force_in_root)
        tool_xpos = self.sim.data.get_body_xpos("Sponge")
        # print("tool, tcp xpos: ", tool_xpos, tcp_xpos)
        tool_xmat_tmp = self.sim.data.get_body_xmat("Sponge")
        # print(tool_xmat, tool_xmat_tmp)
        torque_in_tool = tool_xmat.T.dot(torque_in_root + np.cross((ft_sensor_xpos - tool_xpos), force_in_root))
        self.record_motion.append(tcp_xpos.copy())
        self.record_force.append(force_in_tool.copy())
        self.record_force_root.append(force_in_root.copy())
        self.record_torque.append(torque_in_tool.copy())

        self.data_current_step[0] = self.sim.data.time
        self.data_current_step[1:4] = tcp_xpos.copy()
        current_quat = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(current_quat, tcp_xmat.flatten())
        self.data_current_step[4:8] = current_quat.copy()
        self.data_current_step[15:18] = v_p.copy()
        self.data_current_step[18:21] = v_r.copy()
        self.data_current_step[27:30] = force_in_tool
        self.data_current_step[30:33] = torque_in_tool

        v_p_in_tool = tool_xmat.T.dot(v_p)
        self.record_velocity.append(v_p_in_tool)

        # estimate friction
        friction_estimation = np.zeros(2)
        v_xy = v_p_in_tool[:2]
        f_xy = force_in_tool[:2]
        self.record_friction_norm.append(np.linalg.norm(f_xy))
        self.record_norm_force.append(force_in_tool[2])
        if len(self.record_friction_norm) >= self.friction_estimation_window:
            self.record_friction_norm.pop(0)
            self.record_norm_force.pop(0)

            norm_force = np.array(self.record_norm_force)
            friction_norm = np.array(self.record_friction_norm)
            norm_force_norm = np.linalg.norm(norm_force)
            if norm_force_norm > 0.:
                # self.mu = norm_force.dot(friction_norm) / np.linalg.norm(norm_force)
                mu_tmp = norm_force.dot(friction_norm) / (norm_force_norm ** 2)
                if mu_tmp > 0:
                    self.mu = min(self.mu, mu_tmp)
            friction_estimation = - v_xy * self.mu * force_in_tool[2] / np.linalg.norm(v_xy)

        else:
            if np.linalg.norm(f_xy) > 1.0:
                mu_tmp = - f_xy.dot(v_xy) / force_in_tool[2] / np.linalg.norm(v_xy)
                # if mu_tmp > 0.0:
                #     self.mu = min(self.mu, mu_tmp)
        self.record_mu.append(self.mu)
        self.record_friction.append(friction_estimation)
        # force_in_tool[:2] -= friction_estimation
        # force_in_root = tool_xmat.dot(force_in_tool)

        # time
        dT = self.sim.data.time - self.last_time
        self.last_time = self.sim.data.time
        self.record_time.append(normalizedTime)

        # inverse dynamic control
        # force controller
        #  translational
        target_vel_in_root = np.zeros(6)
        target_vel_in_root[:3] = tool_xmat.dot(target_xyz)
        # target_vel_in_root[3:] = tool_xmat.dot(target_vel_in_tool[3:])
        force_target = self.contact_force_target + \
                       self.force_profile_mode * self.force_mag * np.sin(2 * np.pi * normalizedTime)
        self.record_force_target.append(force_target)
        self.force_pid.update(feedback_value=force_in_tool[2], set_point=force_target,
                              current_time=self.sim.data.time)
        self.target_force_in_tool[2] -= self.force_pid.output

        #  rotational
        fixed_axis = np.zeros(3)
        angle = 0.0
        if np.linalg.norm(force_in_root) > self.min_react_force:
            current_tool_dir = tool_xmat.dot(self.orig_tool_dir)
            tool_y_dir = np.array([0, 1.0, 0])
            proj_force_in_root = force_in_root - force_in_root.dot(tool_y_dir) * tool_y_dir
            desired_tool_dir = proj_force_in_root / np.linalg.norm(proj_force_in_root)

            axis = np.cross(current_tool_dir, desired_tool_dir)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(current_tool_dir.dot(desired_tool_dir))
            sign = 1
            if axis[1] < 0:
                sign = -1
            self.record_angle.append((sign * angle))
            if self.lower_angle <= angle < self.upper_angle:
                # adaptedAngularKp = rot_kp/(1 + np.exp(10 * (angle - np.pi/3)))/(1 + np.exp(10 * (np.pi/5 - angle)))
                adaptedAngularKp = self.rot_kp / (1 + np.exp(10 * (angle - np.pi / 4)))
                angularKp = min(adaptedAngularKp, self.rot_kp)
                self.torque_pid.setKp(angularKp)
                angle = angle - self.lower_angle
                angle *= sign
            else:
                angle = 0.0
                self.torque_pid.setKp(self.rot_kp)
        else:
            self.record_angle.append(angle)
        self.torque_pid.update(feedback_value=angle, set_point=0, current_time=self.sim.data.time)
        self.target_force_in_tool[4] -= self.torque_pid.output

        if self.vmp_velocity_mode:
            self.target_xpos = self.target_xpos + dT * target_vel_in_root[:3]
        else:
            self.target_xpos = target_xyz
        self.record_dmp.append(self.target_xpos.copy())

        # position
        u_task = np.zeros(6)
        u_task[:3] = np.multiply(self.kpos, (self.target_xpos - tcp_xpos)) - \
                     np.multiply(self.dpos, v_p) + \
                     tool_xmat.dot(self.target_force_in_tool[:3])

        # orientation
        if self.vmp_velocity_mode:
            delta_mat = rpy2mat(target_vel_in_root[3:])
            self.target_xmat = delta_mat.dot(self.target_xmat)
        else:
            rot_mat = np.zeros(9)
            mujoco_py.functions.mju_quat2Mat(rot_mat, target_quat)
            self.target_xmat = rot_mat.reshape(3, 3)
        rot_diff_mat = self.target_xmat.dot(np.linalg.inv(tcp_xmat))
        rpy_error = mat2rpy(rot_diff_mat)
        u_task[3:] = np.multiply(self.kori, rpy_error) - \
                     np.multiply(self.dori, v_r) + \
                     tool_xmat.dot(self.target_force_in_tool[3:])

        # calculate torque
        jac_p = self.sim.data.get_site_jacp(self.tcp_name)
        jac_r = self.sim.data.get_site_jacr(self.tcp_name)
        jac = np.zeros((6, self.n_joints))

        jac[0:3, :] = get_actuator_data(jac_p.reshape(3, -1), self.actuator_id_in_order)
        jac[3:, :] = get_actuator_data(jac_r.reshape(3, -1), self.actuator_id_in_order)
        torque = jac.T.dot(u_task)

        # passivity
        self.record_po.append(0)  # po
        self.record_rc.append(0)  # rc
        self.record_pobs.append(0)  # passivity_observer

        # null space control
        if desired_joints is None:
            u_null = 20 * (self.static_nullspace_joints - qpos) - 4 * qvel
        else:
            u_null = 20 * (desired_joints - qpos) - 4 * qvel

        NullMat = np.identity(self.n_joints)
        jacT_pinv = np.linalg.pinv(jac.T, 1e-3)
        NullMat = NullMat - jac.T.dot(jacT_pinv)
        u_null = np.clip(u_null, a_min=-50, a_max=50)
        torque_null = NullMat.dot(u_null)

        # compensation for the nonlinear dynamics, including gravity, friction, coriolis force, etc
        torque_bias = get_actuator_data(self.sim.data.qfrc_bias, self.actuator_id_in_order)

        # disturbance
        force_disturb = np.array([0, 0, 0], dtype=np.float64)
        torque_disturb = np.array([0, 0, 0], dtype=np.float64)
        point = np.array([0, 0, 0], dtype=np.float64)
        body = mujoco_py.functions.mj_name2id(self.sim.model, const.OBJ_BODY, "Hand R Palm")
        qfrc_target = np.zeros(self.n_joints, dtype=np.float64)
        if 4 < self.sim.data.time < 10:
            mujoco_py.functions.mj_applyFT(self.sim.model, self.sim.data, force_disturb, torque_disturb, point, body,
                                           qfrc_target)

        torque_cmd = torque + torque_null + torque_bias + qfrc_target
        torque_cmd = np.clip(torque_cmd, a_min=-70, a_max=70)

        # set command to Mujoco for each joint
        for j, jointName in enumerate(self.arm_joint_name_list):
            self.sim.data.ctrl[self.actuator_ids[jointName]] = torque_cmd[j]

        if normalizedTime == 1:
            self.plot()

        self.data_current_step[8:11] = self.target_xpos.copy()
        current_quat = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(current_quat, self.target_xmat.flatten())
        self.data_current_step[11:15] = current_quat.copy()
        self.data_current_step[21:27] = target_vel_in_root.copy()
        self.record_data.append(self.data_current_step.copy())

        return torque_cmd

    def save_recorded_data(self):
        record_index = ['time',
                        'current_pose_x', 'current_pose_y', 'current_pose_z', 'current_pose_qw', 'current_pose_qx',
                        'current_pose_qy', 'current_pose_qz',
                        'target_pose_x', 'target_pose_y', 'target_pose_z', 'target_pose_qw', 'target_pose_qx',
                        'target_pose_qy', 'target_pose_qz',
                        'current_vel_x', 'current_vel_y', 'current_vel_z', 'current_vel_row', 'current_vel_pitch',
                        'current_vel_yaw',
                        'target_vel_x', 'target_vel_y', 'target_vel_z', 'target_vel_row', 'target_vel_pitch',
                        'target_vel_yaw',
                        'ft_sensor_x', 'ft_sensor_y', 'ft_sensor_z', 'ft_sensor_row', 'ft_sensor_pitch',
                        'ft_sensor_yaw']
        df = pd.DataFrame(data=np.array(self.record_data), columns=record_index)
        df_vel = df[['time', 'current_vel_x', 'current_vel_y', 'current_vel_z', 'current_vel_row', 'current_vel_pitch',
                     'current_vel_yaw']]
        df_ft = df[['time', 'ft_sensor_x', 'ft_sensor_y', 'ft_sensor_z', 'ft_sensor_row', 'ft_sensor_pitch', 'ft_sensor_yaw']]
        df.set_index('time')
        # print(dataFrame)
        csv_target_file = '/home/jianfeng/robot_projects/learning-control/robolab/data/armar6-motion/recorded_wiping_data_from_mujoco_'
        csv_target_file = csv_target_file + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
        # Don't forget to add '.csv' at the end of the path
        if self.enable_write_csv:
            df.to_csv(csv_target_file + '_full.csv', index='time', header=True)
            df_vel.to_csv(csv_target_file + '_velocity.csv', index='time', header=True)
            df_ft.to_csv(csv_target_file + '_ft.csv', index='time', header=True)
            print("save data to ", csv_target_file)
        print("Something is wrong or terminated by user ...")


class TaskSpaceExplicitForceImpedanceAdaptiveFrictionConeController:
    def __init__(self, model, sim, config=None, arm_name="RightArm", desired_joints=None):
        self.sim = sim

        if config is None:
            self.config = DEFAULT_EXPLICIT_FORCE_IMPEDANCE_CONFIG
        else:
            self.config = config

        self.root_xpos = np.zeros(3)
        self.root_xmat = np.identity(3)

        self.restart_high_ctrl = False

        self.root_name = self.config['root_name']
        self.root_xpos = self.sim.data.get_body_xpos(self.root_name)
        self.root_xmat = self.sim.data.get_body_xmat(self.root_name)
        self.root_xvelp = self.sim.data.get_body_xvelp(self.root_name)
        self.root_xvelr = self.sim.data.get_body_xvelr(self.root_name)
        self.ft_sensor_xmat = self.sim.data.get_site_xmat("SITE_FT")
        self.tcp_xpos = self.sim.data.get_site_xpos("Hand R TCP")
        self.tcp_xmat = self.sim.data.get_site_xmat("Hand R TCP")
        self.tcp_pose = np.array([0., 0., 0., 1.0, 0., 0., 0.])

        self.kpos = self.config['kpos']
        self.kori = self.config['kori']
        self.dpos = self.config['dpos']
        self.dori = self.config['dori']

        self.platform_joint_list = self.config['platform_joint_list']
        self.platform_actuator_ids = {joint: model.actuator_name2id(joint + "_position")
                                      for joint in self.platform_joint_list}
        self.hand_comfortable_radius = self.config['hand_comfortable_radius']
        self.platform_vel = self.config['platform_vel']

        self.arm_name = self.config['arm_name']
        self.arm_joint_name_list = self.config['arm_joint_name_list']
        self.arm_joint_config = self.config['arm_joint_config']
        self.tcp_name = self.config['tcp_name']

        self.n_joints = len(self.arm_joint_name_list)
        self.actuator_ids = {joint: model.actuator_name2id(joint + "_motor") for joint in self.arm_joint_name_list}
        self.actuator_id_in_order = [self.actuator_ids[joint_name] for joint_name in self.arm_joint_name_list]

        self.init_hand_joint_angle = self.config['init_hand_joint_angle']
        self.init_arm_joint_angle = self.config['init_arm_joint_angle']
        # print("joint ids:", self.actuator_id_in_order)

        if desired_joints is None:
            self.static_nullspace_joints = self.config['init_arm_joint_angle']
        else:
            self.static_nullspace_joints = desired_joints

        self.hand_name = self.config['hand_name']
        self.hand_joint_list = self.config['hand_joint_list']
        self.hand_joint_config = self.config['hand_joint_config']
        self.tcp_name = self.config['tcp_name']
        self.ft_filter_coeff = self.config['ft_filter_coeff']
        self.vel_filter_coeff = self.config['vel_filter_coeff']
        self.enable_write_csv = self.config['enable_write_csv']

        # force controller configuration
        self.force_kp = self.config['force_kp']
        self.force_ki = self.config['force_ki']
        self.force_kd = self.config['force_kd']
        self.contact_force_target = self.config['contact_force_target']
        self.torque_kp = self.config['torque_kp']
        self.torque_ki = self.config['torque_ki']
        self.torque_kd = self.config['torque_kd']
        self.contact_torque_target = self.config['contact_torque_target']
        self.force_mag = self.config['force_mag']
        self.target_vel_in_tool = self.config['target_vel_in_tool']
        self.orig_tool_dir = self.config['orig_tool_dir']
        self.rot_kp = self.config['rot_kp']
        self.rot_ki = self.config['rot_ki']
        self.rot_kd = self.config['rot_kd']
        self.min_react_force = self.config['min_react_force']
        self.lower_angle = self.config['lower_angle']
        self.upper_angle = self.config['upper_angle']
        self.force_pid = PID(self.force_kp, self.force_ki, self.force_kd, current_time=sim.data.time)
        self.force_pid.setWindup(self.config['force_pid_anti_windup'])
        self.torque_pid = PID(self.torque_kp, self.torque_ki, self.torque_kd, current_time=sim.data.time)
        self.torque_pid.setWindup(self.config['torque_pid_anti_windup'])
        self.lcr_kp = self.config['lcr_kp']
        self.lcr_ki = self.config['lcr_ki']
        self.lcr_kd = self.config['lcr_kd']
        self.lcr_pid = PID(self.lcr_kp, self.lcr_ki, self.lcr_kd, current_time=sim.data.time)
        self.lcr_pid.setWindup(self.config['lcr_pid_anti_windup'])
        self.friction_estimation_window = self.config['friction_estimation_window']
        self.safe_friction_cone_llim = self.config['safe_friction_cone_llim']
        self.mu = 100.0

        self.rot_pid = PID(self.rot_kp, self.rot_ki, self.rot_kd, current_time=sim.data.time)
        self.rot_pid.setWindup(self.config['torque_pid_anti_windup'])

        self.platform_kp = self.config['platform_kp']
        self.platform_ki = self.config['platform_ki']
        self.platform_kd = self.config['platform_kd']
        self.platform_pid = PID(self.platform_kp, self.platform_ki, self.platform_kd, current_time=sim.data.time)
        self.rot_pid.setWindup(self.config['platform_pid_anti_windup'])

        self.last_ft_force = np.zeros(3, dtype="float64")
        self.last_ft_torque = np.zeros(3, dtype="float64")
        self.last_vel = np.zeros(6, dtype="float64")

        # passivity-based controller configuration
        self.force_profile_mode = self.config['force_profile_mode']
        self.force_mag = 5
        self.vmp_velocity_mode = self.config['vmp_velocity_mode']
        self.passive_control_enabled = self.config['passive_control_enabled']
        self.time_window = self.config['time_window']

        self.passivity_observer = 0
        self.rc = 1
        self.vc = 0
        self.freez_counter = 0
        self.passive_counter = 0
        self.frozen = False

        self.record_surface_point = []
        self.record_surface_window = 100

        self.start_plot = 1
        self.first_wipe_round = True

        self.contacted_once = False
        self.making_contact = False
        self.making_contact_counter = 0
        self.loose_contact_ratio = self.config['loose_contact_ratio']
        self.loose_contact_recover_enabled = False
        self.loose_contact_recover_counter = 100
        self.force_control_switch = 1

        self.platform_target = np.zeros(3)

        self.last_u_task = np.zeros(6)
        self.u_task_filter_coeff = self.config['u_task_filter_coeff']

        # for anomaly detection
        self.vel_horizon_length = 1
        self.vel_horizon_list = []
        max_force_limit = self.config['max_force_limit']
        self.force_anomaly_mean = []
        self.force_anomaly_mean_val = np.zeros(3)
        self.force_anomaly_ll = []
        self.force_anomaly_ul = []
        self.force_lower_limit = - max_force_limit * np.ones(3)
        self.force_upper_limit = max_force_limit * np.ones(3)

        np.set_printoptions(threshold=np.inf, edgeitems=3, linewidth=150, formatter=dict(float=lambda x: "%.2g" % x))
        self.setup_plot()
        self.set_init_pose()
        self.first_run()

    def set_vel_horizon_length(self, len):
        self.vel_horizon_length = len

    def set_force_threshold(self, val, threshold):
        self.force_anomaly_mean_val = val.flatten()
        self.force_lower_limit = self.force_anomaly_mean_val - threshold
        self.force_upper_limit = self.force_anomaly_mean_val + threshold

    def set_init_pose(self):
        states = self.sim.get_state()
        for joint_id, joint in enumerate(self.arm_joint_name_list):
            upper_bound = self.arm_joint_config[joint][1]
            lower_bound = self.arm_joint_config[joint][0]
            if upper_bound > 3.14:
                upper_bound = 3.14
            if lower_bound < - 3.14:
                lower_bound = - 3.14
            value = random() * (upper_bound - lower_bound) + lower_bound
            addr = self.sim.model.get_joint_qpos_addr(joint)
            # states.qpos[addr] = value
            states.qpos[addr] = self.init_arm_joint_angle[joint_id]

        for joint_id, joint in enumerate(self.hand_joint_list):
            upper_bound = self.hand_joint_config[joint][1]
            lower_bound = self.hand_joint_config[joint][0]
            if upper_bound > 3.14:
                upper_bound = 3.14
            if lower_bound < - 3.14:
                lower_bound = - 3.14
            value = random() * (upper_bound - lower_bound) + lower_bound
            addr = self.sim.model.get_joint_qpos_addr(joint)
            # states.qpos[addr] = value
            states.qpos[addr] = self.init_hand_joint_angle[joint_id]

        # bottle_init = [-.6, 1.3, .95, 1, 0, 0, 0]
        # addr = self.sim.model.get_joint_qpos_addr("Jbottle")
        # for i in range(addr[1] - addr[0]):
        #     states.qpos[addr[0] + i] = bottle_init[i]

        self.sim.set_state(states)
        print("set initial state done")
        self.sim.forward()

    def setup_plot(self):
        plt.ion()
        fig = plt.figure(figsize=plt.figaspect(1))
        # ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        ax1 = fig.add_subplot(3, 3, 1)
        ax2 = fig.add_subplot(3, 3, 2)
        ax3 = fig.add_subplot(3, 3, 3)
        ax4 = fig.add_subplot(3, 3, 4)
        ax5 = fig.add_subplot(3, 3, 5)
        ax6 = fig.add_subplot(3, 3, 6)
        ax7 = fig.add_subplot(3, 3, 7)
        ax8 = fig.add_subplot(3, 3, 8)
        ax9 = fig.add_subplot(3, 3, 9)
        # ax1.set_title('3d position: dmp and tracking')
        ax1.set_title('task space force command')
        ax2.set_title('force in tool frame')
        ax3.set_title('torque in tool frame')
        ax4.set_title('passive observer')
        ax5.set_title('velocity in tool frame')
        ax6.set_title('Resistance Rc')
        ax7.set_title('friction cone')
        ax7.set_ylim(0, 1.1)
        ax8.set_title('loose contact recover enabled')
        # ax8.set_ylim(0, 1.1)
        ax9.set_title('recovery_torque')

        fig2 = plt.figure()
        # self.ax_pca = fig2.gca(adjustable='box', projection='3d')
        self.ax_pca = fig2.add_subplot(1, 1, 1, projection='3d')
        self.ax_pca.set_xlim(0, 0.5)
        self.ax_pca.set_ylim(0.7, 1.2)
        self.ax_pca.set_zlim(1.04, 1.5)
        self.ax_pca.set_xlabel('X axis')
        self.ax_pca.set_ylabel('Y axis')
        self.ax_pca.set_zlabel('Z axis')
        self.ax_pca.view_init(elev=10., azim=-90)

        plt.grid(True)
        plt.draw()
        plt.pause(0.0005)
        self.axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

        self.record_motion = []
        self.record_dmp = []
        self.record_force = []
        self.record_force_root = []
        self.record_torque = []
        self.record_force_target = []
        self.record_time = []
        self.record_data = []
        self.record_angle = []
        self.record_po = []
        self.record_pobs = []
        self.record_rc = []
        self.record_velocity = []
        self.record_mu = []
        self.record_friction = []
        self.record_friction_norm = []
        self.record_norm_force = []
        self.record_friction_cone = []
        self.record_pca = []
        self.record_flag_make_contact = []
        self.record_flag_loose_contact = []
        self.record_recovery_torque = []
        self.record_rot_pid_torque = []
        self.record_u_task = []
        self.record_platform_vel = []
        self.record_platform_target_vel = []
        self.record_pca_sample_step = 50
        self.data_current_step = np.zeros(33)

    def plot(self):
        # for plotting
        recorded_motion = np.array(self.record_motion)
        recorded_dmp = np.array(self.record_dmp)
        recorded_time = np.array(self.record_time)
        recorded_force = np.array(self.record_force)
        recorded_force_anomaly_mean = np.array(self.force_anomaly_mean)
        recorded_force_anomaly_ll = np.array(self.force_anomaly_ll)
        recorded_force_anomaly_ul = np.array(self.force_anomaly_ul)
        recorded_force_root = np.array(self.record_force_root)
        recorded_torque = np.array(self.record_torque)
        recorded_po = np.array(self.record_po)
        recorded_pos = np.array(self.record_pobs)
        recorded_rc = np.array(self.record_rc)
        recorded_angle = np.array(self.record_angle)
        recorded_force_target = np.array(self.record_force_target)
        recorded_velocity = np.array(self.record_velocity)
        recorded_mu = np.array(self.record_mu)
        recorded_friction = np.array(self.record_friction)
        recorded_friction_cone = np.array(self.record_friction_cone)
        recorded_friction_norm = np.array(self.record_friction_norm)
        recorded_norm_force = np.array(self.record_norm_force)
        recorded_pca = np.array(self.record_pca)
        recorded_recovery_torque = np.array(self.record_recovery_torque)
        recorded_u_task = np.array(self.record_u_task)
        # if print_once:
        if self.start_plot == 0:
            # self.axes[0].plot3D(recorded_motion[:, 0], recorded_motion[:, 1], recorded_motion[:, 2], color="orange")
            # self.axes[0].plot3D(recorded_dmp[:, 0], recorded_dmp[:, 1], recorded_dmp[:, 2], color="blue")
            self.axes[0].plot(recorded_time, recorded_u_task[:, 0], color='orange', alpha=0.4, label="u_task_x")
            self.axes[0].plot(recorded_time, recorded_u_task[:, 1], color='green', alpha=0.4, label="u_task_y")
            self.axes[0].plot(recorded_time, recorded_u_task[:, 2], color='blue', alpha=0.4, label="u_task_z")
            self.axes[0].plot(recorded_time, recorded_u_task[:, 3], color='yellow', alpha=0.4, label="u_task_rx")
            self.axes[0].plot(recorded_time, recorded_u_task[:, 4], color='black', alpha=0.4, label="u_task_ry")
            self.axes[0].plot(recorded_time, recorded_u_task[:, 5], color='magenta', alpha=0.4, label="u_task_rz")

            self.axes[1].clear()
            self.axes[1].plot(recorded_time, recorded_force[:, 0], color='orange', alpha=0.4, label="force_tool_x")
            self.axes[1].plot(recorded_time, recorded_force[:, 1], color='green', alpha=0.4, label="force_tool_y")
            self.axes[1].plot(recorded_time, recorded_force[:, 2], color='blue', alpha=0.4, label="force_tool_z")
            self.axes[1].plot(recorded_time, recorded_force_anomaly_mean[:, 0], color='orange', alpha=0.4, label="force_anomaly_x", linestyle='--')
            self.axes[1].plot(recorded_time, recorded_force_anomaly_mean[:, 1], color='green', alpha=0.4, label="force_anomaly_y", linestyle='--')
            self.axes[1].plot(recorded_time, recorded_force_anomaly_mean[:, 2], color='blue', alpha=0.4, label="force_anomaly_z", linestyle='--')
            self.axes[1].fill_between(recorded_time, recorded_force_anomaly_ll[:, 0], recorded_force_anomaly_ul[:, 0], where=recorded_force_anomaly_ul[:, 0] >= recorded_force_anomaly_ll[:, 0], facecolor='orange', interpolate=True, alpha=0.2)
            self.axes[1].fill_between(recorded_time, recorded_force_anomaly_ll[:, 1], recorded_force_anomaly_ul[:, 1], where=recorded_force_anomaly_ul[:, 1] >= recorded_force_anomaly_ll[:, 1], facecolor='green', interpolate=True, alpha=0.2)
            self.axes[1].fill_between(recorded_time, recorded_force_anomaly_ll[:, 2], recorded_force_anomaly_ul[:, 2], where=recorded_force_anomaly_ul[:, 2] >= recorded_force_anomaly_ll[:, 2], facecolor='blue', interpolate=True, alpha=0.2)
            # self.axes[1].plot(recorded_time, recorded_friction[:, 0], color='red', alpha=0.2, linestyle=":", label="friction_tool_x")
            # self.axes[1].plot(recorded_time, recorded_friction[:, 1], color='green', alpha=0.2, linestyle=":", label="friction_tool_y")
            self.axes[1].plot(recorded_time, recorded_force_target, color='black', alpha=0.4, label="target_z")
            self.axes[2].clear()
            self.axes[2].plot(recorded_time, recorded_torque[:, 0], color='orange', alpha=0.4, label="torque_tool_x")
            self.axes[2].plot(recorded_time, recorded_torque[:, 1], color='green', alpha=0.4, label="torque_tool_y")
            self.axes[2].plot(recorded_time, recorded_torque[:, 2], color='blue', alpha=0.4, label="torque_tool_z")

            self.axes[3].plot(recorded_time, recorded_po, alpha=0.4)
            # self.axes[4].plot(recorded_time, recorded_pos, alpha=0.4)
            # self.axes[4].plot(recorded_time, recorded_force_root[:, 0], color='orange', alpha=0.4, label="force_root_x")
            # self.axes[4].plot(recorded_time, recorded_force_root[:, 1], color='green', alpha=0.4, label="force_root_y")
            # self.axes[4].plot(recorded_time, recorded_force_root[:, 2], color='blue', alpha=0.4, label="force_root_z")
            self.axes[4].clear()
            self.axes[4].plot(recorded_time, recorded_velocity[:, 0], color='orange', alpha=0.4)
            self.axes[4].plot(recorded_time, recorded_velocity[:, 1], color='green', alpha=0.4)
            self.axes[4].plot(recorded_time, recorded_velocity[:, 2], color='blue', alpha=0.4)
            # self.axes[5].plot(recorded_norm_force, recorded_friction_norm, alpha=0.4)
            # self.axes[5].plot(recorded_time, recorded_rc, alpha=0.4)
            # self.axes[5].plot(recorded_time, recorded_angle, alpha=0.4)
            self.axes[5].plot(recorded_time, self.record_platform_target_vel, color='orange', alpha=0.4)
            self.axes[5].plot(recorded_time, self.record_platform_vel, color='cyan', alpha=0.4)

            self.axes[6].plot(recorded_time, recorded_friction_cone, alpha=0.4, label="friction cone")
            self.axes[7].clear()
            # self.axes[7].plot(recorded_time, self.record_flag_loose_contact, alpha=0.4)
            self.axes[7].plot(recorded_time, self.record_rot_pid_torque, color='orange', alpha=0.4)
            # self.axes[8].plot(recorded_time, self.record_flag_make_contact, alpha=0.4)
            self.axes[8].clear()
            self.axes[8].plot(recorded_time, recorded_recovery_torque[:, 0], color='orange', alpha=0.4)
            self.axes[8].plot(recorded_time, recorded_recovery_torque[:, 1], color='green', alpha=0.4)
            self.axes[8].plot(recorded_time, recorded_recovery_torque[:, 2], color='blue', alpha=0.4)


            # [0-8] components, 3 weighted vectors, [9, 10, 11] tcp position
            self.ax_pca.clear()
            self.ax_pca.plot3D(recorded_motion[:, 0], recorded_motion[:, 1], recorded_motion[:, 2], alpha=0.5, color='cyan')
            self.ax_pca.quiver(recorded_pca[:, 9], recorded_pca[:, 10], recorded_pca[:, 11], recorded_pca[:, 0], recorded_pca[:, 1], recorded_pca[:, 2], alpha=0.9, color='orange', arrow_length_ratio=0.1)
            self.ax_pca.quiver(recorded_pca[:, 9], recorded_pca[:, 10], recorded_pca[:, 11], recorded_pca[:, 3], recorded_pca[:, 4], recorded_pca[:, 5], alpha=0.9, color='green', arrow_length_ratio=0.1)
            self.ax_pca.quiver(recorded_pca[:, 9], recorded_pca[:, 10], recorded_pca[:, 11], recorded_pca[:, 6], recorded_pca[:, 7], recorded_pca[:, 8], alpha=0.5, color='blue', arrow_length_ratio=0.1)
            # self.ax_pca.scatter(recorded_pca[:, 0], recorded_pca[:, 1], recorded_pca[:, 2], alpha=0.4, color='orange')
            # self.ax_pca.scatter(recorded_pca[:, 3], recorded_pca[:, 4], recorded_pca[:, 5], alpha=0.4, color='green')
            # self.ax_pca.scatter(recorded_pca[:, 6], recorded_pca[:, 7], recorded_pca[:, 8], alpha=0.4, color='blue')
            self.ax_pca.set_xlabel('X axis')
            self.ax_pca.set_ylabel('Y axis')
            self.ax_pca.set_zlabel('Z axis')
        else:
            self.start_plot -= 1

        self.axes[0].set_title('task space force command')
        self.axes[1].set_title('force in tool frame')
        self.axes[2].set_title('torque in tool frame')
        self.axes[3].set_title('passive observer')
        self.axes[4].set_title('velocity in tool frame')
        self.axes[5].set_title('Platform distance')
        self.axes[6].set_title('friction cone')
        self.axes[7].set_title('accum. torque cmd')
        self.axes[8].set_title('recovery_torque')

        plt.draw()
        plt.pause(0.0001)
        # plt.pause(1)

    def clear_plot_variable(self):
        self.record_time.clear()
        self.record_force.clear()
        self.record_force_root.clear()
        self.record_force_target.clear()
        self.record_torque.clear()
        self.record_po.clear()
        self.record_pobs.clear()
        self.record_rc.clear()
        self.record_angle.clear()
        self.record_velocity.clear()
        self.record_mu.clear()
        self.record_friction.clear()
        self.record_friction_cone.clear()
        self.record_pca.clear()
        self.record_flag_loose_contact.clear()
        self.record_flag_make_contact.clear()
        self.record_recovery_torque.clear()
        self.record_rot_pid_torque.clear()
        self.record_u_task.clear()
        self.record_platform_vel.clear()
        self.record_platform_target_vel.clear()
        self.force_anomaly_mean.clear()
        self.force_anomaly_ll.clear()
        self.force_anomaly_ul.clear()
        # self.record_friction_norm.clear()

    def first_run(self):
        self.start_time = self.sim.data.time
        self.last_time = self.sim.data.time

        self.init_tcp_xpos = self.tcp_xpos.copy()
        self.tcp_pose[:3] = self.tcp_xpos.copy()
        mujoco_py.functions.mju_mat2Quat(self.tcp_pose[3:], self.tcp_xmat.flatten())

        self.tool_to_hand_xmat = self.tcp_xmat.copy().T
        # self.tool_xmat = np.identity(3)
        self.tool_dir = np.array([0, 0, 1])

        self.ft_force_init = -get_sensor_value(self.sim, 3, "Force_R")
        self.ft_torque_init = -get_sensor_value(self.sim, 3, "Torque_R")
        print("init force torque: ", self.ft_force_init, self.ft_torque_init)

        # self.last_tcp_xpos_target = tcp_xpos.copy()

        self.target_xpos = self.tcp_xpos.copy()
        self.target_xmat = self.tcp_xmat.copy()

        self.target_force_in_tool = np.zeros(6)

    def read(self, targets):

        if 'position' in targets.keys():
            target_xyz = targets['position']
        else:
            print('target is invalid, because key word {} is missing'.format('position'))

        if 'orientation' in targets.keys():
            target_quat = targets['orientation']
        else:
            print('target is invalid, because key word {} is missing'.format('orientation'))

        if 'nullspace' in targets.keys():
            desired_joints = targets['nullspace']
        else:
            desired_joints = None

        if 'canonical_value' in targets.keys():
            canonical_value = targets['canonical_value']
        else:
            canonical_value = None

        return target_xyz, target_quat, desired_joints, canonical_value

    def to_root_xpos(self, xpos):
        return xpos - self.root_xpos

    def to_root_xmat(self, xmat):
        return self.root_xmat.T.dot(xmat)

    def control(self, targets):
        target_xyz, target_quat, desired_joints, normalizedTime = self.read(targets)
        self.root_xvelp = self.sim.data.get_body_xvelp(self.root_name)
        self.root_xvelr = self.sim.data.get_body_xvelr(self.root_name)
        target_xyz -= self.root_xvelp
        # velr not calculated yet !!

        self.force_anomaly_mean.append(self.force_anomaly_mean_val)
        self.force_anomaly_ll.append(self.force_lower_limit)
        self.force_anomaly_ul.append(self.force_upper_limit)

        # position, velocity feedback
        tcp_xpos = self.to_root_xpos(self.tcp_xpos)
        tcp_xmat = self.to_root_xmat(self.tcp_xmat)
        self.tcp_pose[:3] = tcp_xpos.copy()
        mujoco_py.functions.mju_mat2Quat(self.tcp_pose[3:], self.tcp_xmat.flatten())

        v_p = self.sim.data.get_site_xvelp(self.tcp_name).copy()
        v_r = self.sim.data.get_site_xvelr(self.tcp_name).copy()
        v_p = self.vel_filter_coeff * v_p + (1 - self.vel_filter_coeff) * self.last_vel[:3]
        v_r = self.vel_filter_coeff * v_r + (1 - self.vel_filter_coeff) * self.last_vel[3:]
        self.last_vel[:3] = v_p.copy()
        self.last_vel[3:] = v_r.copy()
        qpos = get_actuator_data(self.sim.data.qpos, self.actuator_id_in_order)
        qvel = get_actuator_data(self.sim.data.qvel, self.actuator_id_in_order)
        # print("qpos: ", qpos)

        # force feedback
        ft_force = -get_sensor_value(self.sim, 3, "Force_R") - self.ft_force_init
        ft_torque = -get_sensor_value(self.sim, 3, "Torque_R") - self.ft_torque_init
        ft_force = self.ft_filter_coeff * ft_force + (1 - self.ft_filter_coeff) * self.last_ft_force
        ft_torque = self.ft_filter_coeff * ft_torque + (1 - self.ft_filter_coeff) * self.last_ft_torque
        self.last_ft_force = ft_force.copy()
        self.last_ft_torque = ft_torque.copy()

        ft_sensor_xpos = self.sim.data.get_site_xpos("SITE_FT")
        ft_sensor_xmat = self.sim.data.get_site_xmat("SITE_FT")
        tool_xmat = tcp_xmat.dot(self.tool_to_hand_xmat)
        force_in_root = ft_sensor_xmat.dot(ft_force)
        torque_in_root = ft_sensor_xmat.dot(ft_torque)
        force_in_tool = tool_xmat.T.dot(force_in_root)
        tool_xpos = self.sim.data.get_site_xpos("Tool Frame")
        tool_xmat_tmp = self.sim.data.get_site_xmat("Tool Frame")
        torque_in_tool = tool_xmat.T.dot(torque_in_root + np.cross((ft_sensor_xpos - tool_xpos), force_in_root))
        friction_cone_torque = self.mu * self.contact_force_target * 0.09
        print("friction_cone_torque: ", friction_cone_torque)
        for i in range(3):
            if -friction_cone_torque < torque_in_tool[i] < friction_cone_torque:
                torque_in_tool[i] = 0.0
            else:
                torque_in_tool[i] = torque_in_tool[i] - np.sign(torque_in_tool[i]) * friction_cone_torque

        self.record_motion.append(tcp_xpos.copy())
        self.record_force.append(force_in_tool.copy())
        self.record_force_root.append(force_in_root.copy())
        self.record_torque.append(torque_in_tool.copy())

        self.data_current_step[0] = self.sim.data.time
        self.data_current_step[1:4] = tcp_xpos.copy()
        current_quat = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(current_quat, tcp_xmat.flatten())
        self.data_current_step[4:8] = current_quat.copy()
        self.data_current_step[15:18] = v_p.copy()
        self.data_current_step[18:21] = v_r.copy()
        self.data_current_step[27:30] = force_in_tool
        self.data_current_step[30:33] = torque_in_tool

        # self.vel_horizon_list.append(np.concatenate((v_p.copy(), v_r.copy())))
        self.vel_horizon_list.append(v_p.copy())
        if len(self.vel_horizon_list) > self.vel_horizon_length:
            self.vel_horizon_list.pop(0)

        v_p_in_tool = tool_xmat.T.dot(v_p)
        self.record_velocity.append(v_p_in_tool)

        # estimate friction
        friction_estimation = np.zeros(2)
        v_xy = v_p_in_tool[:2]
        f_xy = force_in_tool[:2]
        if not self.first_wipe_round:
            if np.linalg.norm(v_p) > 0.001:
                self.record_friction_norm.append(np.linalg.norm(f_xy))
                self.record_norm_force.append(force_in_tool[2])
            if len(self.record_friction_norm) >= self.friction_estimation_window:
                self.record_friction_norm.pop(0)
                self.record_norm_force.pop(0)

                norm_force = np.array(self.record_norm_force)
                friction_norm = np.array(self.record_friction_norm)
                norm_force_norm = np.linalg.norm(norm_force)
                if norm_force_norm > 0.:
                    mu_tmp = norm_force.dot(friction_norm) / (norm_force_norm ** 2)
                    if mu_tmp > 0:
                        self.mu = min(self.mu, mu_tmp)
                friction_estimation = - v_xy * self.mu * force_in_tool[2] / np.linalg.norm(v_xy)

            else:
                if np.linalg.norm(f_xy) > 1.0:
                    mu_tmp = - f_xy.dot(v_xy) / force_in_tool[2] / np.linalg.norm(v_xy)
                    # if mu_tmp > 0.0:
                    #     self.mu = min(self.mu, mu_tmp)

        self.mu = max(self.mu, self.safe_friction_cone_llim)
        self.record_mu.append(self.mu)
        self.record_friction.append(friction_estimation)
        # force_in_tool[:2] -= friction_estimation
        # force_in_root = tool_xmat.dot(force_in_tool)

        # time
        dT = self.sim.data.time - self.last_time
        self.last_time = self.sim.data.time
        self.record_time.append(normalizedTime)

        # pca plane detector
        if np.linalg.norm(force_in_root) > 6.0:
            # mu_tmp = max(np.tan(np.pi/7.0), self.mu * 0.995)
            # self.mu = mu_tmp
            self.record_surface_point.append(tcp_xpos.copy())
            if len(self.record_surface_point) == self.record_surface_window:
                self.record_surface_point.pop(0)
                pca = PCA(n_components=3)
                pca.fit(np.array(self.record_surface_point))
                weighted_components = pca.components_ * pca.singular_values_ * 0.1
                # self.record_pca.append(weighted_components.flatten())
                if self.record_pca_sample_step >= 50:
                    self.record_pca_sample_step = 0
                    self.record_pca.append(np.concatenate((weighted_components.T, tcp_xpos), axis=None))
                self.record_pca_sample_step += 1
                # print("components: ", pca.components_, " \nsingular values: ", pca.singular_values_, "\nweighted pca:", weighted_components)

        # inverse dynamic control
        # force controller
        #  translational
        target_vel_in_root = np.zeros(6)
        target_vel_in_root[:3] = tool_xmat.dot(target_xyz)
        # target_vel_in_root[3:] = tool_xmat.dot(target_vel_in_tool[3:])
        force_target = self.contact_force_target + \
                       self.force_profile_mode * self.force_mag * np.sin(2 * np.pi * normalizedTime)
        self.record_force_target.append(force_target)
        self.force_pid.update(feedback_value=force_in_tool[2], set_point=force_target,
                              current_time=self.sim.data.time)

        #  rotational
        friction_cone = np.arctan(self.mu)
        self.record_friction_cone.append(friction_cone)
        fixed_axis = np.zeros(3)
        angle = 0.0
        if np.linalg.norm(force_in_root) > self.min_react_force:
            current_tool_dir = tool_xmat.dot(self.orig_tool_dir)
            tool_y_dir = np.array([0, 1.0, 0])
            proj_force_in_root = force_in_root - force_in_root.dot(tool_y_dir) * tool_y_dir
            desired_tool_dir = proj_force_in_root / np.linalg.norm(proj_force_in_root)

            axis = np.cross(current_tool_dir, desired_tool_dir)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(current_tool_dir.dot(desired_tool_dir))
            sign = 1
            if axis[1] < 0:
                sign = -1
            self.record_angle.append((sign * angle))
            if friction_cone <= angle < self.upper_angle:
                adapted_angular_kp = self.rot_kp / (1 + np.exp(10 * (angle - np.pi / 4)))
                angular_kp = min(adapted_angular_kp, self.rot_kp)
                self.rot_pid.setKp(angular_kp)
                angle = angle - friction_cone
                angle *= sign
            else:
                angle = 0.0
                self.rot_pid.setKp(self.rot_kp)
        else:
            self.record_angle.append(angle)
        self.rot_pid.update(feedback_value=angle, set_point=0, current_time=self.sim.data.time)
        self.torque_pid.update(feedback_value=torque_in_tool[1], set_point=0, current_time=self.sim.data.time)

        # loose contact detector
        if force_in_tool[2] >= self.loose_contact_ratio * self.contact_force_target:
            self.making_contact_counter += 1
            if self.making_contact_counter > 20:
                self.making_contact = True
                self.loose_contact_recover_enabled = False
            else:
                self.making_contact = False

        if not self.contacted_once and self.making_contact:
            self.contacted_once = True

        self.record_flag_make_contact.append(self.making_contact)

        compensate_axis = np.zeros(3)
        if self.contacted_once and abs(v_p_in_tool[2]) > 0.2 and friction_cone < 1.0:
            self.making_contact_counter = 0
            v = np.zeros(3)  # for now, only consider the vel in x direction (tool frame)
            v[0] = v_p_in_tool[0]
            compensate_axis = np.cross(np.array([0, 0, self.contact_force_target]), v)
            norm = np.linalg.norm(compensate_axis)
            if norm > 0:
                compensate_axis /= norm
            else:
                compensate_axis.fill(0)
            self.force_control_switch *= 0.5
            self.loose_contact_recover_enabled = True
            # self.restart_high_ctrl = True
            self.loose_contact_recover_counter -= 1
        else:
            self.force_control_switch = 1
            self.restart_high_ctrl = False

        self.record_flag_loose_contact.append(self.loose_contact_recover_enabled)

        recovery_torque = np.zeros(3)
        v_in_force_dir = 0
        # if self.loose_contact_recover_enabled:
        if self.loose_contact_recover_counter < 100:
            v_in_force_dir = abs(v_p_in_tool[2])
            self.loose_contact_recover_counter -= 1
            if self.loose_contact_recover_counter == 0:
                self.loose_contact_recover_counter = 100

        self.lcr_pid.update(feedback_value=v_in_force_dir, set_point=0, current_time=self.sim.data.time)
        recovery_torque = self.lcr_pid.output * compensate_axis
        self.target_force_in_tool[3:] += recovery_torque
        self.record_recovery_torque.append(recovery_torque)

        # fetch force control results
        self.target_force_in_tool[2] -= self.force_pid.output * self.force_control_switch
        self.target_force_in_tool[4] -= self.rot_pid.output
        self.record_rot_pid_torque.append(self.target_force_in_tool[4])
        self.target_force_in_tool[4] -= self.torque_pid.output

        # calculating target position based on vmp velocity
        if self.vmp_velocity_mode:
            self.target_xpos = self.target_xpos + dT * target_vel_in_root[:3]
        else:
            self.target_xpos = target_xyz
        self.record_dmp.append(self.target_xpos.copy())

        # position
        u_task = np.zeros(6)
        u_task[:3] = np.multiply(self.kpos, (self.target_xpos - tcp_xpos)) - \
                     np.multiply(self.dpos, v_p) + \
                     tool_xmat.dot(self.target_force_in_tool[:3])

        # orientation
        if self.vmp_velocity_mode:
            delta_mat = rpy2mat(target_vel_in_root[3:])
            self.target_xmat = delta_mat.dot(self.target_xmat)
        else:
            rot_mat = np.zeros(9)
            mujoco_py.functions.mju_quat2Mat(rot_mat, target_quat)
            self.target_xmat = rot_mat.reshape(3, 3)
        rot_diff_mat = self.target_xmat.dot(np.linalg.inv(tcp_xmat))
        rpy_error = mat2rpy(rot_diff_mat)
        u_task[3:] = np.multiply(self.kori, rpy_error) - \
                     np.multiply(self.dori, v_r) + \
                     tool_xmat.dot(self.target_force_in_tool[3:])

        u_task = self.u_task_filter_coeff * u_task + (1 - self.u_task_filter_coeff) * self.last_u_task
        self.last_u_task = u_task.copy()
        self.record_u_task.append(u_task.copy())

        # calculate torque
        jac_p = self.sim.data.get_site_jacp(self.tcp_name)
        jac_r = self.sim.data.get_site_jacr(self.tcp_name)
        jac = np.zeros((6, self.n_joints))

        jac[0:3, :] = get_actuator_data(jac_p.reshape(3, -1), self.actuator_id_in_order)
        jac[3:, :] = get_actuator_data(jac_r.reshape(3, -1), self.actuator_id_in_order)
        torque = jac.T.dot(u_task)

        # passivity
        self.record_po.append(0)  # po
        self.record_rc.append(0)  # rc
        self.record_pobs.append(0)  # passivity_observer

        # null space control
        if desired_joints is None:
            u_null = 20 * (self.static_nullspace_joints - qpos) - 4 * qvel
        else:
            u_null = 20 * (desired_joints - qpos) - 4 * qvel

        nullspace_projection = np.identity(self.n_joints)
        jac_t_pinv = np.linalg.pinv(jac.T, 1e-3)
        nullspace_projection = nullspace_projection - jac.T.dot(jac_t_pinv)
        u_null = np.clip(u_null, a_min=-50, a_max=50)
        torque_null = nullspace_projection.dot(u_null)

        # compensation for the nonlinear dynamics, including gravity, friction, coriolis force, etc
        torque_bias = get_actuator_data(self.sim.data.qfrc_bias, self.actuator_id_in_order)

        # disturbance
        force_disturb = np.array([0, 0, 0], dtype=np.float64)
        torque_disturb = np.array([0, 0, 0], dtype=np.float64)
        point = np.array([0, 0, 0], dtype=np.float64)
        body = mujoco_py.functions.mj_name2id(self.sim.model, const.OBJ_BODY, "Hand R Palm")
        qfrc_target = np.zeros(self.n_joints, dtype=np.float64)
        if 4 < self.sim.data.time < 10:
            mujoco_py.functions.mj_applyFT(self.sim.model, self.sim.data, force_disturb, torque_disturb, point, body,
                                           qfrc_target)

        torque_cmd = torque + torque_null + torque_bias + qfrc_target
        torque_cmd = np.clip(torque_cmd, a_min=-70, a_max=70)

        # platform controller
        dist = tcp_xpos[0] - self.init_tcp_xpos[0]
        if dist < -self.hand_comfortable_radius:
            dist += self.hand_comfortable_radius/2.0
        elif dist > self.hand_comfortable_radius:
            dist -= self.hand_comfortable_radius/2.0
        else:
            dist = 0
        self.platform_pid.update(feedback_value=dist, set_point=0.0, current_time=self.sim.data.time)
        self.platform_target[0] += (self.platform_vel[0] - self.platform_pid.output) * dT
        # print('platform target: ', self.platform_target, self.platform_pid.output, dist)
        self.record_platform_target_vel.append(self.platform_target[0])
        self.record_platform_vel.append(self.root_xvelp)
        print("platform vel p:", self.root_xvelp)

        # set command to Mujoco for each joint
        for j, jointName in enumerate(self.arm_joint_name_list):
            self.sim.data.ctrl[self.actuator_ids[jointName]] = torque_cmd[j]
        for j, jointName in enumerate(self.platform_joint_list):
            self.sim.data.ctrl[self.platform_actuator_ids[jointName]] = self.platform_target[j]

        if normalizedTime == 1:
            self.first_wipe_round = False
            self.plot()
            self.clear_plot_variable()

        self.data_current_step[8:11] = self.target_xpos.copy()
        current_quat = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(current_quat, self.target_xmat.flatten())
        self.data_current_step[11:15] = current_quat.copy()
        self.data_current_step[21:27] = target_vel_in_root.copy()
        self.record_data.append(self.data_current_step.copy())

        return torque_cmd

    def save_recorded_data(self):
        record_index = ['time',
                        'current_pose_x', 'current_pose_y', 'current_pose_z', 'current_pose_qw', 'current_pose_qx',
                        'current_pose_qy', 'current_pose_qz',
                        'target_pose_x', 'target_pose_y', 'target_pose_z', 'target_pose_qw', 'target_pose_qx',
                        'target_pose_qy', 'target_pose_qz',
                        'current_vel_x', 'current_vel_y', 'current_vel_z', 'current_vel_row', 'current_vel_pitch',
                        'current_vel_yaw',
                        'target_vel_x', 'target_vel_y', 'target_vel_z', 'target_vel_row', 'target_vel_pitch',
                        'target_vel_yaw',
                        'ft_sensor_x', 'ft_sensor_y', 'ft_sensor_z', 'ft_sensor_row', 'ft_sensor_pitch',
                        'ft_sensor_yaw']
        df = pd.DataFrame(data=np.array(self.record_data), columns=record_index)
        df.set_index('time')
        # print(dataFrame)
        csv_target_file = '/home/jianfeng/robot_projects/learning-control/robolab/data/armar6-motion/recorded_wiping_data_from_mujoco_'
        csv_target_file = csv_target_file + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S') + '.csv'
        # Don't forget to add '.csv' at the end of the path
        if self.enable_write_csv:
            df.to_csv(csv_target_file, index='time', header=True)
            print("save data to ", csv_target_file)
        print("Something is wrong or terminated ...")


class TaskSpaceVelBasedExplicitForceImpedanceAdaptiveFrictionConeController:
    def __init__(self, model, sim, config=None, arm_name="RightArm", desired_joints=None):
        self.sim = sim

        if config is None:
            self.config = DEFAULT_VEL_BASED_EXPLICIT_FORCE_IMPEDANCE_CONFIG
        else:
            self.config = config

        self.restart_high_ctrl = False

        self.kpos = self.config['kpos']
        self.kori = self.config['kori']
        self.dpos = self.config['dpos']
        self.dori = self.config['dori']

        self.arm_name = self.config['arm_name']
        self.arm_joint_name_list = self.config['arm_joint_name_list']
        self.arm_joint_config = self.config['arm_joint_config']
        self.tcp_name = self.config['tcp_name']

        self.n_joints = len(self.arm_joint_name_list)
        self.actuator_ids = {joint: model.actuator_name2id(joint + "_motor") for joint in self.arm_joint_name_list}
        self.actuator_id_in_order = [self.actuator_ids[joint_name] for joint_name in self.arm_joint_name_list]
        self.init_hand_joint_angle = self.config['init_hand_joint_angle']
        self.init_arm_joint_angle = self.config['init_arm_joint_angle']
        # print("joint ids:", self.actuator_id_in_order)

        if desired_joints is None:
            self.static_nullspace_joints = self.config['init_arm_joint_angle']
        else:
            self.static_nullspace_joints = desired_joints

        self.hand_name = self.config['hand_name']
        self.hand_joint_list = self.config['hand_joint_list']
        self.hand_joint_config = self.config['hand_joint_config']
        self.tcp_name = self.config['tcp_name']
        self.ft_filter_coeff = self.config['ft_filter_coeff']
        self.vel_filter_coeff = self.config['vel_filter_coeff']
        self.enable_write_csv = self.config['enable_write_csv']

        # force controller configuration
        self.force_kp = self.config['force_kp']
        self.force_ki = self.config['force_ki']
        self.force_kd = self.config['force_kd']
        self.contact_force_target = self.config['contact_force_target']
        self.torque_kp = self.config['torque_kp']
        self.torque_ki = self.config['torque_ki']
        self.torque_kd = self.config['torque_kd']
        self.contact_torque_target = self.config['contact_torque_target']
        self.force_mag = self.config['force_mag']
        # self.target_vel_in_tool = self.config['target_vel_in_tool']
        self.orig_tool_dir = self.config['orig_tool_dir']
        self.rot_kp = self.config['rot_kp']
        self.rot_ki = self.config['rot_ki']
        self.rot_kd = self.config['rot_kd']
        self.min_react_force = self.config['min_react_force']
        self.lower_angle = self.config['lower_angle']
        self.upper_angle = self.config['upper_angle']
        self.force_pid = PID(self.force_kp, self.force_ki, self.force_kd, current_time=sim.data.time)
        self.force_pid.setWindup(self.config['force_pid_anti_windup'])
        self.torque_pid = PID(self.torque_kp, self.torque_ki, self.torque_kd, current_time=sim.data.time)
        self.torque_pid.setWindup(self.config['torque_pid_anti_windup'])
        self.friction_estimation_window = self.config['friction_estimation_window']
        self.safe_friction_cone_llim = self.config['safe_friction_cone_llim']
        self.mu = 100.0

        self.rot_pid = PID(self.rot_kp, self.rot_ki, self.rot_kd, current_time=sim.data.time)
        self.rot_pid.setWindup(self.config['torque_pid_anti_windup'])

        self.last_ft_force = np.zeros(3, dtype="float64")
        self.last_ft_torque = np.zeros(3, dtype="float64")
        self.last_vel = np.zeros(6, dtype="float64")

        # passivity-based controller configuration
        self.force_profile_mode = self.config['force_profile_mode']
        self.force_mag = 5
        self.vmp_velocity_mode = self.config['vmp_velocity_mode']
        self.passive_control_enabled = self.config['passive_control_enabled']
        self.time_window = self.config['time_window']

        self.passivity_observer = 0
        self.rc = 1
        self.vc = 0
        self.freez_counter = 0
        self.passive_counter = 0
        self.frozen = False

        self.record_surface_point = []
        self.record_surface_window = 100

        self.start_plot = 1
        self.first_wipe_round = True

        self.contacted_once = False
        self.making_contact = False
        self.making_contact_counter = 0
        self.loose_contact_ratio = self.config['loose_contact_ratio']

        np.set_printoptions(threshold=np.inf, edgeitems=3, linewidth=150, formatter=dict(float=lambda x: "%.2g" % x))
        self.setup_plot()
        self.set_init_pose()
        self.first_run()

    def set_init_pose(self):
        states = self.sim.get_state()
        for joint_id, joint in enumerate(self.arm_joint_name_list):
            upper_bound = self.arm_joint_config[joint][1]
            lower_bound = self.arm_joint_config[joint][0]
            if upper_bound > 3.14:
                upper_bound = 3.14
            if lower_bound < - 3.14:
                lower_bound = - 3.14
            value = random() * (upper_bound - lower_bound) + lower_bound
            addr = self.sim.model.get_joint_qpos_addr(joint)
            # states.qpos[addr] = value
            states.qpos[addr] = self.init_arm_joint_angle[joint_id]

        for joint_id, joint in enumerate(self.hand_joint_list):
            upper_bound = self.hand_joint_config[joint][1]
            lower_bound = self.hand_joint_config[joint][0]
            if upper_bound > 3.14:
                upper_bound = 3.14
            if lower_bound < - 3.14:
                lower_bound = - 3.14
            value = random() * (upper_bound - lower_bound) + lower_bound
            addr = self.sim.model.get_joint_qpos_addr(joint)
            # states.qpos[addr] = value
            states.qpos[addr] = self.init_hand_joint_angle[joint_id]

        # bottle_init = [-.6, 1.3, .95, 1, 0, 0, 0]
        # addr = self.sim.model.get_joint_qpos_addr("Jbottle")
        # for i in range(addr[1] - addr[0]):
        #     states.qpos[addr[0] + i] = bottle_init[i]

        self.sim.set_state(states)
        print("set initial state done")
        self.sim.forward()

    def setup_plot(self):
        plt.ion()
        fig = plt.figure(figsize=plt.figaspect(1))
        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        ax2 = fig.add_subplot(3, 3, 2)
        ax3 = fig.add_subplot(3, 3, 3)
        ax4 = fig.add_subplot(3, 3, 4)
        ax5 = fig.add_subplot(3, 3, 5)
        ax6 = fig.add_subplot(3, 3, 6)
        ax7 = fig.add_subplot(3, 3, 7)
        ax8 = fig.add_subplot(3, 3, 8)
        ax9 = fig.add_subplot(3, 3, 9)
        ax1.set_title('3d position: dmp and tracking')
        ax2.set_title('force in tool frame')
        ax3.set_title('torque in tool frame')
        ax4.set_title('passive observer')
        ax5.set_title('passive observer moving mean')
        ax6.set_title('Resistance Rc')
        ax7.set_title('friction cone')
        ax7.set_ylim(0, 0.8)
        ax8.set_title('tmp')
        ax8.set_ylim(0, 1.0)
        ax9.set_title('tmp')

        fig2 = plt.figure()
        # self.ax_pca = fig2.gca(adjustable='box', projection='3d')
        self.ax_pca = fig2.add_subplot(1, 1, 1, projection='3d')
        self.ax_pca.set_xlim(0, 0.5)
        self.ax_pca.set_ylim(0.7, 1.2)
        self.ax_pca.set_zlim(1.04, 1.2)

        plt.grid(True)
        plt.draw()
        plt.pause(0.0005)
        self.axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

        self.record_motion = []
        self.record_dmp = []
        self.record_force = []
        self.record_force_root = []
        self.record_torque = []
        self.record_force_target = []
        self.record_time = []
        self.record_data = []
        self.record_angle = []
        self.record_po = []
        self.record_pobs = []
        self.record_rc = []
        self.record_velocity = []
        self.record_mu = []
        self.record_friction = []
        self.record_friction_norm = []
        self.record_norm_force = []
        self.record_friction_cone = []
        self.record_pca = []
        self.record_pca_sample_step = 50
        self.data_current_step = np.zeros(33)

    def plot(self):
        # for plotting
        recorded_motion = np.array(self.record_motion)
        recorded_dmp = np.array(self.record_dmp)
        recorded_time = np.array(self.record_time)
        recorded_force = np.array(self.record_force)
        recorded_force_root = np.array(self.record_force_root)
        recorded_torque = np.array(self.record_torque)
        recorded_po = np.array(self.record_po)
        recorded_pos = np.array(self.record_pobs)
        recorded_rc = np.array(self.record_rc)
        recorded_angle = np.array(self.record_angle)
        recorded_force_target = np.array(self.record_force_target)
        recorded_velocity = np.array(self.record_velocity)
        recorded_mu = np.array(self.record_mu)
        recorded_friction = np.array(self.record_friction)
        recorded_friction_cone = np.array(self.record_friction_cone)
        recorded_friction_norm = np.array(self.record_friction_norm)
        recorded_norm_force = np.array(self.record_norm_force)
        recorded_pca = np.array(self.record_pca)
        # if print_once:
        if self.start_plot == 0:
            self.axes[0].plot3D(recorded_motion[:, 0], recorded_motion[:, 1], recorded_motion[:, 2], color="orange")
            self.axes[0].plot3D(recorded_dmp[:, 0], recorded_dmp[:, 1], recorded_dmp[:, 2], color="blue")
            self.axes[1].plot(recorded_time, recorded_force[:, 0], color='red', alpha=0.4, label="force_tool_x")
            self.axes[1].plot(recorded_time, recorded_force[:, 1], color='green', alpha=0.4, label="force_tool_y")
            self.axes[1].plot(recorded_time, recorded_force[:, 2], color='blue', alpha=0.4, label="force_tool_z")
            # self.axes[1].plot(recorded_time, recorded_friction[:, 0], color='red', alpha=0.2, linestyle=":", label="friction_tool_x")
            # self.axes[1].plot(recorded_time, recorded_friction[:, 1], color='green', alpha=0.2, linestyle=":", label="friction_tool_y")
            self.axes[1].plot(recorded_time, recorded_force_target, color='yellow', alpha=0.4, label="target_z")
            self.axes[2].plot(recorded_time, recorded_torque[:, 0], color='orange', alpha=0.4, label="torque_tool_x")
            self.axes[2].plot(recorded_time, recorded_torque[:, 1], color='green', alpha=0.4, label="torque_tool_y")
            self.axes[2].plot(recorded_time, recorded_torque[:, 2], color='blue', alpha=0.4, label="torque_tool_z")

            self.axes[3].plot(recorded_time, recorded_po, alpha=0.4)
            # self.axes[4].plot(recorded_time, recorded_pos, alpha=0.4)
            # self.axes[4].plot(recorded_time, recorded_force_root[:, 0], color='orange', alpha=0.4, label="force_root_x")
            # self.axes[4].plot(recorded_time, recorded_force_root[:, 1], color='green', alpha=0.4, label="force_root_y")
            # self.axes[4].plot(recorded_time, recorded_force_root[:, 2], color='blue', alpha=0.4, label="force_root_z")
            self.axes[4].plot(recorded_time, recorded_velocity[:, 0], color='red', alpha=0.4)
            self.axes[4].plot(recorded_time, recorded_velocity[:, 1], color='green', alpha=0.4)
            self.axes[4].plot(recorded_time, recorded_velocity[:, 2], color='blue', alpha=0.4)
            # self.axes[5].plot(recorded_norm_force, recorded_friction_norm, alpha=0.4)
            # self.axes[5].plot(recorded_time, recorded_rc, alpha=0.4)
            self.axes[5].plot(recorded_time, recorded_angle, alpha=0.4)

            self.axes[6].plot(recorded_time, recorded_friction_cone, alpha=0.4, label="friction cone")
            self.axes[7].plot(recorded_time, recorded_mu, alpha=0.4)
            origin = np.zeros_like(recorded_pca[:, 0])
            # [0-8] components, 3 weighted vectors, [9, 10, 11] tcp position
            self.ax_pca.plot3D(recorded_motion[:, 0], recorded_motion[:, 1], recorded_motion[:, 2], alpha=0.5, color='cyan')
            self.ax_pca.quiver(recorded_pca[:, 9], recorded_pca[:, 10], recorded_pca[:, 11], recorded_pca[:, 0], recorded_pca[:, 1], recorded_pca[:, 2], alpha=0.9, color='orange', arrow_length_ratio=0.1)
            self.ax_pca.quiver(recorded_pca[:, 9], recorded_pca[:, 10], recorded_pca[:, 11], recorded_pca[:, 3], recorded_pca[:, 4], recorded_pca[:, 5], alpha=0.9, color='green', arrow_length_ratio=0.1)
            self.ax_pca.quiver(recorded_pca[:, 9], recorded_pca[:, 10], recorded_pca[:, 11], recorded_pca[:, 6], recorded_pca[:, 7], recorded_pca[:, 8], alpha=0.5, color='blue', arrow_length_ratio=0.1)
            # self.ax_pca.scatter(recorded_pca[:, 0], recorded_pca[:, 1], recorded_pca[:, 2], alpha=0.4, color='orange')
            # self.ax_pca.scatter(recorded_pca[:, 3], recorded_pca[:, 4], recorded_pca[:, 5], alpha=0.4, color='green')
            # self.ax_pca.scatter(recorded_pca[:, 6], recorded_pca[:, 7], recorded_pca[:, 8], alpha=0.4, color='blue')
        else:
            self.start_plot -= 1

        self.record_time.clear()
        self.record_force.clear()
        self.record_force_root.clear()
        self.record_force_target.clear()
        self.record_torque.clear()
        self.record_po.clear()
        self.record_pobs.clear()
        self.record_rc.clear()
        self.record_angle.clear()
        self.record_velocity.clear()
        self.record_mu.clear()
        self.record_friction.clear()
        self.record_friction_cone.clear()
        self.record_pca.clear()
        # self.record_friction_norm.clear()
        plt.draw()
        plt.pause(0.0001)

    def first_run(self):
        self.start_time = self.sim.data.time
        self.last_time = self.sim.data.time

        ft_sensor_xmat = self.sim.data.get_site_xmat("SITE_FT")
        tcp_xpos = self.sim.data.get_site_xpos("Hand R TCP")
        tcp_xmat = self.sim.data.get_site_xmat("Hand R TCP")
        self.tool_to_hand_xmat = tcp_xmat.copy().T
        # self.tool_xmat = np.identity(3)
        self.tool_dir = np.array([0, 0, 1])

        self.ft_force_init = -get_sensor_value(self.sim, 3, "Force_R")
        self.ft_torque_init = -get_sensor_value(self.sim, 3, "Torque_R")

        # self.last_tcp_xpos_target = tcp_xpos.copy()

        self.target_xpos = tcp_xpos.copy()
        self.target_xmat = tcp_xmat.copy()

        # self.target_force_in_tool = np.zeros(6)
        self.target_vel_in_tool = np.zeros(6)

    def read(self, targets):

        if 'position' in targets.keys():
            target_xyz = targets['position']
        else:
            print('target is invalid, because key word {} is missing'.format('position'))

        if 'orientation' in targets.keys():
            target_quat = targets['orientation']
        else:
            print('target is invalid, because key word {} is missing'.format('orientation'))

        if 'nullspace' in targets.keys():
            desired_joints = targets['nullspace']
        else:
            desired_joints = None

        if 'canonical_value' in targets.keys():
            canonical_value = targets['canonical_value']
        else:
            canonical_value = None

        return target_xyz, target_quat, desired_joints, canonical_value

    def control(self, targets):
        target_xyz, target_quat, desired_joints, normalizedTime = self.read(targets)
        # print("ctl: ", target_xyz, target_quat, normalizedTime)

        # position, velocity feedback
        tcp_xpos = self.sim.data.get_site_xpos(self.tcp_name).copy()
        tcp_xmat = self.sim.data.get_site_xmat(self.tcp_name).copy()
        v_p = self.sim.data.get_site_xvelp(self.tcp_name).copy()
        v_r = self.sim.data.get_site_xvelr(self.tcp_name).copy()
        v_p = self.vel_filter_coeff * v_p + (1 - self.vel_filter_coeff) * self.last_vel[:3]
        v_r = self.vel_filter_coeff * v_r + (1 - self.vel_filter_coeff) * self.last_vel[3:]
        self.last_vel[:3] = v_p.copy()
        self.last_vel[3:] = v_r.copy()
        qpos = get_actuator_data(self.sim.data.qpos, self.actuator_id_in_order)
        qvel = get_actuator_data(self.sim.data.qvel, self.actuator_id_in_order)
        # print("qpos: ", qpos)

        # force feedback
        ft_force = -get_sensor_value(self.sim, 3, "Force_R") - self.ft_force_init
        ft_torque = -get_sensor_value(self.sim, 3, "Torque_R") - self.ft_torque_init
        ft_force = self.ft_filter_coeff * ft_force + (1 - self.ft_filter_coeff) * self.last_ft_force
        ft_torque = self.ft_filter_coeff * ft_torque + (1 - self.ft_filter_coeff) * self.last_ft_torque
        for i in range(3):
            if -0.1 < ft_torque[i] < 0.1:
                ft_torque[i] = 0.0
            else:
                ft_torque[i] = ft_torque[i] - np.sign(ft_torque[i]) * 0.1

        self.last_ft_force = ft_force.copy()
        self.last_ft_torque = ft_torque.copy()
        ft_sensor_xpos = self.sim.data.get_site_xpos("SITE_FT")
        ft_sensor_xmat = self.sim.data.get_site_xmat("SITE_FT")
        tool_xmat = tcp_xmat.dot(self.tool_to_hand_xmat)
        force_in_root = ft_sensor_xmat.dot(ft_force)
        torque_in_root = ft_sensor_xmat.dot(ft_torque)
        force_in_tool = tool_xmat.T.dot(force_in_root)
        tool_xpos = self.sim.data.get_body_xpos("Sponge")
        tool_xmat_tmp = self.sim.data.get_body_xmat("Sponge")

        torque_in_tool = tool_xmat.T.dot(torque_in_root + np.cross((ft_sensor_xpos - tool_xpos), force_in_root))
        self.record_motion.append(tcp_xpos.copy())
        self.record_force.append(force_in_tool.copy())
        self.record_force_root.append(force_in_root.copy())
        self.record_torque.append(torque_in_tool.copy())

        self.data_current_step[0] = self.sim.data.time
        self.data_current_step[1:4] = tcp_xpos.copy()
        current_quat = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(current_quat, tcp_xmat.flatten())
        self.data_current_step[4:8] = current_quat.copy()
        self.data_current_step[15:18] = v_p.copy()
        self.data_current_step[18:21] = v_r.copy()
        self.data_current_step[27:30] = force_in_tool
        self.data_current_step[30:33] = torque_in_tool

        v_p_in_tool = tool_xmat.T.dot(v_p)
        self.record_velocity.append(v_p_in_tool)

        # estimate friction
        friction_estimation = np.zeros(2)
        v_xy = v_p_in_tool[:2]
        f_xy = force_in_tool[:2]
        if not self.first_wipe_round:
            if np.linalg.norm(v_p) > 0.001:
                self.record_friction_norm.append(np.linalg.norm(f_xy))
                self.record_norm_force.append(force_in_tool[2])
            if len(self.record_friction_norm) >= self.friction_estimation_window:
                self.record_friction_norm.pop(0)
                self.record_norm_force.pop(0)

                norm_force = np.array(self.record_norm_force)
                friction_norm = np.array(self.record_friction_norm)
                norm_force_norm = np.linalg.norm(norm_force)
                if norm_force_norm > 0.:
                    mu_tmp = norm_force.dot(friction_norm) / (norm_force_norm ** 2)
                    if mu_tmp > 0:
                        self.mu = min(self.mu, mu_tmp)
                friction_estimation = - v_xy * self.mu * force_in_tool[2] / np.linalg.norm(v_xy)

        self.mu = max(self.mu, self.safe_friction_cone_llim)
        self.record_mu.append(self.mu)
        self.record_friction.append(friction_estimation)
        # force_in_tool[:2] -= friction_estimation
        # force_in_root = tool_xmat.dot(force_in_tool)

        # time
        dT = self.sim.data.time - self.last_time
        self.last_time = self.sim.data.time
        self.record_time.append(normalizedTime)

        # pca plane detector
        if np.linalg.norm(force_in_root) > 6.0:
            # mu_tmp = max(np.tan(np.pi/7.0), self.mu * 0.995)
            # self.mu = mu_tmp
            self.record_surface_point.append(tcp_xpos.copy())
            if len(self.record_surface_point) == self.record_surface_window:
                self.record_surface_point.pop(0)
                pca = PCA(n_components=3)
                pca.fit(np.array(self.record_surface_point))
                weighted_components = pca.components_ * pca.singular_values_ * 0.1
                # self.record_pca.append(weighted_components.flatten())
                if self.record_pca_sample_step >= 50:
                    self.record_pca_sample_step = 0
                    self.record_pca.append(np.concatenate((weighted_components.T, tcp_xpos), axis=None))
                self.record_pca_sample_step += 1
                print("components: ", pca.components_, " \nsingular values: ", pca.singular_values_, "\nweighted pca:", weighted_components)

        # inverse dynamic control
        # force controller
        #  translational
        target_vel_in_root = np.zeros(6)
        force_target = self.contact_force_target + \
                       self.force_profile_mode * self.force_mag * np.sin(2 * np.pi * normalizedTime)
        self.record_force_target.append(force_target)
        self.force_pid.update(feedback_value=force_in_tool[2], set_point=force_target, current_time=self.sim.data.time)
        self.target_vel_in_tool[2] -= self.force_pid.output
        target_vel_in_root[:3] = tool_xmat.dot((self.target_vel_in_tool[:3] + target_xyz))

        #  rotational
        friction_cone = np.arctan(self.mu)
        self.record_friction_cone.append(friction_cone)
        fixed_axis = np.zeros(3)
        angle = 0.0
        if np.linalg.norm(force_in_root) > self.min_react_force:
            current_tool_dir = tool_xmat.dot(self.orig_tool_dir)
            tool_y_dir = np.array([0, 1.0, 0])
            proj_force_in_root = force_in_root - force_in_root.dot(tool_y_dir) * tool_y_dir
            desired_tool_dir = proj_force_in_root / np.linalg.norm(proj_force_in_root)

            axis = np.cross(current_tool_dir, desired_tool_dir)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(current_tool_dir.dot(desired_tool_dir))
            sign = 1
            if axis[1] < 0:
                sign = -1
            self.record_angle.append((sign * angle))
            # if friction_cone <= angle < self.upper_angle:
            if self.lower_angle <= angle < self.upper_angle:
                adapted_angular_kp = self.rot_kp / (1 + np.exp(10 * (angle - np.pi / 4)))
                angular_kp = min(adapted_angular_kp, self.rot_kp)
                self.rot_pid.setKp(angular_kp)
                # angle = angle - friction_cone
                angle = angle - self.lower_angle
                angle *= sign
            else:
                angle = 0.0
                self.rot_pid.setKp(self.rot_kp)
        else:
            self.record_angle.append(angle)
        self.rot_pid.update(feedback_value=angle, set_point=0, current_time=self.sim.data.time)
        self.target_vel_in_tool[4] -= self.rot_pid.output

        self.torque_pid.update(feedback_value=torque_in_tool[1], set_point=0, current_time=self.sim.data.time)
        # self.target_vel_in_tool[4] -= self.rot_pid.output
        target_vel_in_root[3:] = tool_xmat.dot(self.target_vel_in_tool[3:])

        # loose contact detector
        if force_in_tool[2] >= self.loose_contact_ratio * self.contact_force_target:
            self.making_contact_counter += 1
            if self.making_contact_counter > 20:
                self.making_contact = True
            else:
                self.making_contact = False
        else:
            self.making_contact_counter = 0
            compensate_axis = np.cross(np.array([0, 0, self.contact_force_target]), v_p_in_tool)
            compensate_axis /= np.linalg.norm(compensate_axis)

        if not self.contacted_once and self.making_contact:
            self.contacted_once = True

        # if self.contacted_once and

        # calculating target position based on vmp velocity
        if self.vmp_velocity_mode:
            self.target_xpos = self.target_xpos + dT * target_vel_in_root[:3]
        else:
            self.target_xpos = target_xyz
        self.record_dmp.append(self.target_xpos.copy())

        # position
        u_task = np.zeros(6)
        u_task[:3] = np.multiply(self.kpos, (self.target_xpos - tcp_xpos)) - np.multiply(self.dpos, v_p)

        # orientation
        if self.vmp_velocity_mode:
            delta_mat = rpy2mat(target_vel_in_root[3:] * dT)
            self.target_xmat = delta_mat.dot(self.target_xmat)
        else:
            rot_mat = np.zeros(9)
            mujoco_py.functions.mju_quat2Mat(rot_mat, target_quat)
            self.target_xmat = rot_mat.reshape(3, 3)
        rot_diff_mat = self.target_xmat.dot(np.linalg.inv(tcp_xmat))
        rpy_error = mat2rpy(rot_diff_mat)
        u_task[3:] = np.multiply(self.kori, rpy_error) - np.multiply(self.dori, v_r)

        # calculate torque
        jac_p = self.sim.data.get_site_jacp(self.tcp_name)
        jac_r = self.sim.data.get_site_jacr(self.tcp_name)
        jac = np.zeros((6, self.n_joints))

        jac[0:3, :] = get_actuator_data(jac_p.reshape(3, -1), self.actuator_id_in_order)
        jac[3:, :] = get_actuator_data(jac_r.reshape(3, -1), self.actuator_id_in_order)
        torque = jac.T.dot(u_task)

        # passivity
        self.record_po.append(0)  # po
        self.record_rc.append(0)  # rc
        self.record_pobs.append(0)  # passivity_observer

        # null space control
        if desired_joints is None:
            u_null = 20 * (self.static_nullspace_joints - qpos) - 4 * qvel
        else:
            u_null = 20 * (desired_joints - qpos) - 4 * qvel

        nullspace_projection = np.identity(self.n_joints)
        jac_t_pinv = np.linalg.pinv(jac.T, 1e-3)
        nullspace_projection = nullspace_projection - jac.T.dot(jac_t_pinv)
        u_null = np.clip(u_null, a_min=-50, a_max=50)
        torque_null = nullspace_projection.dot(u_null)

        # compensation for the nonlinear dynamics, including gravity, friction, coriolis force, etc
        torque_bias = get_actuator_data(self.sim.data.qfrc_bias, self.actuator_id_in_order)

        # disturbance
        force_disturb = np.array([0, 0, 0], dtype=np.float64)
        torque_disturb = np.array([0, 0, 0], dtype=np.float64)
        point = np.array([0, 0, 0], dtype=np.float64)
        body = mujoco_py.functions.mj_name2id(self.sim.model, const.OBJ_BODY, "Hand R Palm")
        qfrc_target = np.zeros(self.n_joints, dtype=np.float64)
        if 4 < self.sim.data.time < 10:
            mujoco_py.functions.mj_applyFT(self.sim.model, self.sim.data, force_disturb, torque_disturb, point, body,
                                           qfrc_target)

        torque_cmd = torque + torque_null + torque_bias + qfrc_target
        torque_cmd = np.clip(torque_cmd, a_min=-70, a_max=70)

        # set command to Mujoco for each joint
        for j, jointName in enumerate(self.arm_joint_name_list):
            self.sim.data.ctrl[self.actuator_ids[jointName]] = torque_cmd[j]

        if normalizedTime == 1:
            self.first_wipe_round = False
            self.plot()

        self.data_current_step[8:11] = self.target_xpos.copy()
        current_quat = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(current_quat, self.target_xmat.flatten())
        self.data_current_step[11:15] = current_quat.copy()
        self.data_current_step[21:27] = target_vel_in_root.copy()
        self.record_data.append(self.data_current_step.copy())

        return torque_cmd

    def save_recorded_data(self):
        record_index = ['time',
                        'current_pose_x', 'current_pose_y', 'current_pose_z', 'current_pose_qw', 'current_pose_qx',
                        'current_pose_qy', 'current_pose_qz',
                        'target_pose_x', 'target_pose_y', 'target_pose_z', 'target_pose_qw', 'target_pose_qx',
                        'target_pose_qy', 'target_pose_qz',
                        'current_vel_x', 'current_vel_y', 'current_vel_z', 'current_vel_row', 'current_vel_pitch',
                        'current_vel_yaw',
                        'target_vel_x', 'target_vel_y', 'target_vel_z', 'target_vel_row', 'target_vel_pitch',
                        'target_vel_yaw',
                        'ft_sensor_x', 'ft_sensor_y', 'ft_sensor_z', 'ft_sensor_row', 'ft_sensor_pitch',
                        'ft_sensor_yaw']
        df = pd.DataFrame(data=np.array(self.record_data), columns=record_index)
        df.set_index('time')
        # print(dataFrame)
        csv_target_file = '/home/jianfeng/robot_projects/learning-control/robolab/data/armar6-motion/recorded_wiping_data_from_mujoco_'
        csv_target_file = csv_target_file + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S') + '.csv'
        # Don't forget to add '.csv' at the end of the path
        if self.enable_write_csv:
            df.to_csv(csv_target_file, index='time', header=True)
            print("save data to ", csv_target_file)
        print("Something is wrong or terminated ...")

